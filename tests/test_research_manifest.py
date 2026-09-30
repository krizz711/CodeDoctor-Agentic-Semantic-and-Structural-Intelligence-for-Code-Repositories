"""Run with python3 -m unittest discover -s tests -p test_research_manifest.py."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import research_manifest as manifest


class ResearchManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_hashes_large_file_and_detects_content_change(self):
        path = self.root / "result.json"
        payload = b"0123456789" * 300000
        path.write_bytes(payload)
        first = manifest.hash_file(path)
        self.assertEqual(first["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(first["size_bytes"], len(payload))
        path.write_bytes(b"new result")
        self.assertNotEqual(first["sha256"], manifest.hash_file(path)["sha256"])

    def test_change_during_hashing_is_rejected(self):
        path = self.root / "result.json"
        path.write_bytes(b"result")
        actual_fstat = manifest.os.fstat
        calls = []

        def changing_fstat(descriptor):
            calls.append(descriptor)
            if len(calls) == 2:
                path.write_bytes(b"changed result")
            return actual_fstat(descriptor)

        with mock.patch.object(manifest.os, "fstat", side_effect=changing_fstat):
            with self.assertRaisesRegex(manifest.ManifestError, "changed while hashing"):
                manifest.hash_file(path)

    def test_missing_or_directory_input_is_rejected(self):
        for path in (self.root / "missing", self.root):
            with self.subTest(path=path):
                with self.assertRaises(manifest.ManifestError):
                    manifest.hash_file(path)

    def test_json_publication_refuses_overwrite_and_leaves_no_temp(self):
        output = self.root / "manifest.json"
        manifest.write_manifest({"unicode": "café"}, output)
        original = output.read_bytes()
        self.assertEqual(json.loads(original), {"unicode": "café"})
        with self.assertRaisesRegex(manifest.ManifestError, "Refusing to overwrite"):
            manifest.write_manifest({"changed": True}, output)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_publication_failure_leaves_no_partial_output(self):
        output = self.root / "manifest.json"
        with mock.patch.object(manifest.os, "link", side_effect=OSError("unsupported")):
            with self.assertRaisesRegex(manifest.ManifestError, "hard links"):
                manifest.write_manifest({"complete": True}, output)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_git_failure_and_timeout_are_useful_errors(self):
        failures = [
            (FileNotFoundError(), "Git is required"),
            (subprocess.TimeoutExpired("git", 10), "exceeded 10 seconds"),
            (subprocess.CalledProcessError(128, "git", stderr="not a git repository"),
             "not a git repository"),
        ]
        for failure, message in failures:
            with self.subTest(failure=failure):
                with mock.patch.object(manifest.subprocess, "run", side_effect=failure):
                    with self.assertRaisesRegex(manifest.ManifestError, message):
                        manifest.git_metadata(self.root)

    def test_unavailable_memory_is_null(self):
        with mock.patch.object(manifest.os, "sysconf", create=True,
                               side_effect=ValueError("not supported")):
            self.assertIsNone(manifest.physical_memory_bytes())

    def test_cli_collects_real_git_state_without_executing_declared_command(self):
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(self.root), *args], check=True,
                capture_output=True, text=True, encoding="utf-8", timeout=10,
            ).stdout.strip()

        git("init")
        config = self.root / "config.yaml"
        config.write_text("seed: 42\n", encoding="utf-8")
        git("add", "config.yaml")
        git("-c", "user.name=Manifest Test", "-c", "user.email=manifest@example.invalid",
            "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.root / 'no-hooks'}",
            "commit", "-m", "Test fixture")
        expected_head = git("rev-parse", "HEAD")
        self.assertFalse(manifest.git_metadata(self.root)["dirty"])
        result = self.root / "result.json"
        result.write_text('{"score": 0.5}\n', encoding="utf-8")
        self.assertTrue(manifest.git_metadata(self.root)["dirty"])
        output = self.root / "manifest.json"
        sentinel = self.root / "must-not-exist"
        command = f"touch '{sentinel}'"
        script = Path(manifest.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, str(script), "--repo", str(self.root),
             "--config", str(config), "--artifact", str(result),
             "--artifact", str(config), "--output", str(output),
             "--command", command, "--seed", "42"],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        recorded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(recorded["git"]["head"], expected_head)
        self.assertTrue(recorded["git"]["dirty"])
        self.assertEqual(recorded["inputs"]["config"], manifest.hash_file(config))
        self.assertEqual(recorded["inputs"]["artifacts"],
                         [manifest.hash_file(result), manifest.hash_file(config)])
        self.assertEqual(recorded["declared_run"]["command"], command)
        self.assertEqual(recorded["declared_run"]["seed"], 42)
        self.assertFalse(recorded["declared_run"]["verified_by_this_tool"])
        self.assertFalse(recorded["declared_run"]["execution_performed_by_this_tool"])
        self.assertFalse(sentinel.exists())
        self.assertEqual(datetime.fromisoformat(recorded["created_at_utc"])
                         .utcoffset().total_seconds(), 0)
        self.assertIn("python_version", recorded["environment"])
        self.assertIn("host_physical_memory_bytes", recorded["environment"])
        self.assertIsInstance(recorded["environment"]["installed_distributions"], list)

    def test_cli_missing_input_never_creates_manifest(self):
        output = self.root / "manifest.json"
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_context:
            manifest.main(["--output", str(output), "--config", str(self.root / "absent")])
        self.assertEqual(exit_context.exception.code, 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
