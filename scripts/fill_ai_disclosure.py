"""Fill supported facts in the retained official AI disclosure form.

Only word/document.xml is patched. All template fields, styles, relationships
and opaque package parts remain present. Human attestations and signatures are
never generated. Request summaries are explicitly not quoted historical prompts.
This utility uses only Python's standard library.
"""
from __future__ import annotations

import argparse
import re
import tempfile
from copy import copy
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_SHA256 = "b3947950332a44f27c85655ed1c29f295a6b9313ce8eb7c9994bc93506e2069e"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PARAGRAPH = re.compile(r"<w:p(?:\s[^>]*?)?(?:/>|>.*?</w:p>)", re.DOTALL)
TEXT = re.compile(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", re.DOTALL)


def patch_paragraph(paragraph: str, text: str) -> str:
    """Replace displayed text, retaining the original paragraph/run XML."""
    first = True

    def replace(match: re.Match[str]) -> str:
        nonlocal first
        value = escape(text) if first else ""
        first = False
        return match[1] + value + match[3]

    result = TEXT.sub(replace, paragraph)
    if first:
        raise ValueError("the official template text slot is missing")
    return result


def keep_with_next(paragraph: str) -> str:
    """Keep a feature's labels with its description without changing its style."""
    if "<w:pPr>" in paragraph:
        return paragraph.replace("<w:pPr>", "<w:pPr><w:keepNext/>", 1)
    opening = re.match(r"<w:p(?:\s[^>]*)?>", paragraph)
    if opening is None:
        raise ValueError("feature label paragraph is missing")
    return paragraph[:opening.end()] + "<w:pPr><w:keepNext/></w:pPr>" + paragraph[opening.end():]


def build_disclosure(template: Path, output: Path, *, team: str = "Doctors",
                     project: str = "CodeDoctor", institution: str =
                     "Vellore Institute of Technology Vellore", overwrite: bool = False) -> dict:
    template, output = template.resolve(), output.resolve()
    if template == output:
        raise ValueError("the retained template and output must be different paths")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite after reviewing the output path")
    raw = template.read_bytes()
    if sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("template differs from the original official form at f89c969; inspect it before editing")
    with ZipFile(template) as source:
        document = source.read("word/document.xml").decode("utf-8")
        root = ET.fromstring(document)
        paragraphs = root.find(f"{W}body").findall(f"{W}p")
        texts = ["".join(p.itertext()) for p in paragraphs]
        if len(paragraphs) != 41 or len(PARAGRAPH.findall(document)) != 41:
            raise ValueError("official form paragraph structure changed")
        expected = {3: "Team Name:", 9: "Did your team use", 22: "1. Feature Name",
                    26: "1. Feature Name", 32: "AI usage complies", 33: "No proprietary",
                    36: "Name of Team Representative:", 38: "Signature:"}
        for index, prefix in expected.items():
            if not texts[index].startswith(prefix):
                raise ValueError(f"official field {index} no longer has its expected label")

        replacements = {
            3: f"Team Name: {team}",
            4: f"Project / Product Name: {project}",
            5: f"Organization / Institution (if any): {institution}",
            9: "Did your team use any Artificial Intelligence (AI) in developing this project? Yes",
            10: ("OpenAI Codex assisted with the current repository review, code and documentation "
                 "corrections, evidence checks and disclosure preparation. Earlier project records "
                 "also name Claude and Antigravity IDE; exact versions, feature attribution and "
                 "human modifications require team confirmation. This form does not certify "
                 "those earlier attributions or supply a representative's consent."),
            13: "Idea generation / brainstorming: AI-assisted retrieval and verification design review.",
            14: "Code generation or assistance: Repository fixes, scripts and test assistance; attribution review pending.",
            15: "UI / UX design: Earlier records report workbench assistance; detailed attribution pending.",
            16: "Content creation: Technical documentation, submission materials and disclosure draft.",
            17: "Data analysis: Inspection of raw evaluation evidence and synthetic fixture limitations.",
            18: "Testing / debugging: Test execution and failure diagnosis; checks are recorded separately.",
            19: "Other: Dependency, configuration and reproducibility review.",
            22: "1. Feature Name: Retrieval pipeline and repository workbench",
            23: "2. Self-Generated/AI-Generated/Both: Team classification pending",
            24: ("3. Description (AI tools/platform Used, prompt used, Output Summary, Modification): "
                 "Earlier records name Claude, Antigravity IDE and OpenAI Codex. Request summary "
                 "(not a verbatim prompt): assist code retrieval, caching, verification and the "
                 "workbench. Output summary: the repository contains these implementations and "
                 "tests. Exact prompts and feature-specific authorship are not established by "
                 "the records inspected here. Human modifications and final classification must "
                 "be completed by the team."),
            26: "1. Feature Name: Current submission corrections and disclosure generation",
            27: "2. Self-Generated/AI-Generated/Both: AI-assisted; team classification pending",
            28: ("3. Description (AI tools/platform Used, prompt used, Output Summary, Modification): "
                 "AI tools: OpenAI Codex; local openai/whisper-tiny.en ASR through Transformers. "
                 "Request summary (not a verbatim prompt): review requirements, correct claims, "
                 "preserve evidence and the form, and prepare a shorter demo. Output summary: "
                 "local code, docs, dependency and disclosure corrections; provisional transcript "
                 "and timestamps from local ASR and an FFmpeg cut-only video preview. No audio "
                 "was uploaded for ASR. The transcript, cuts, narration, human modifications and "
                 "final acceptance require team review."),
            29: "",
            32: "AI usage complies with guidelines and policies. Pending human confirmation",
            33: "No proprietary or copyrighted data misused. Pending human confirmation",
        }
        index = 0

        def update(match: re.Match[str]) -> str:
            nonlocal index
            paragraph = match[0]
            if index in replacements:
                paragraph = patch_paragraph(paragraph, replacements[index])
            if index in {21, 22, 23, 26, 27}:
                paragraph = keep_with_next(paragraph)
            index += 1
            return paragraph

        updated = PARAGRAPH.sub(update, document).encode("utf-8")
        checked = ET.fromstring(updated)
        if len(checked.find(f"{W}body").findall(f"{W}p")) != 41:
            raise ValueError("generated form lost an official field")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".docx", delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            with ZipFile(temp_path, "w") as target:
                for item in source.infolist():
                    target.writestr(copy(item), updated if item.filename == "word/document.xml"
                                    else source.read(item.filename))
            with ZipFile(temp_path) as target:
                changed = [name for name in source.namelist() if source.read(name) != target.read(name)]
                if changed != ["word/document.xml"]:
                    raise ValueError(f"unexpected template package changes: {changed}")
            temp_path.replace(output)
        finally:
            temp_path.unlink(missing_ok=True)
    return {"output": str(output), "template_sha256": TEMPLATE_SHA256,
            "sha256": sha256(output.read_bytes()).hexdigest(), "human_signoff": "pending"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "build/disclosure/template.docx")
    parser.add_argument("--output", type=Path, default=ROOT / "LangAI3.0_AI_Disclosure.docx")
    parser.add_argument("--team", default="Doctors")
    parser.add_argument("--project", default="CodeDoctor")
    parser.add_argument("--institution", default="Vellore Institute of Technology Vellore")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_disclosure(args.template, args.output, team=args.team, project=args.project,
                                  institution=args.institution, overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Disclosure generation failed: {error}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
