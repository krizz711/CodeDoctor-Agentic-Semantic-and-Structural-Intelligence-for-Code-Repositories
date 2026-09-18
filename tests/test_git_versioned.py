import random
import subprocess
from pathlib import Path
import pytest

from app.indexing.versioned import GitVersionManager, VersionedIndex
from app.retrieval.hybrid import Document


def run_git_cmd(repo_dir: Path, args: list[str]) -> str:
    res = subprocess.run(
        ["git"] + args,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


@pytest.fixture(scope="module")
def git_test_repo(tmp_path_factory):
    repo_dir = tmp_path_factory.mktemp("git_repo")

    # Initialize git repo with config
    run_git_cmd(repo_dir, ["init", "-b", "main"])
    run_git_cmd(repo_dir, ["config", "user.name", "TestDev"])
    run_git_cmd(repo_dir, ["config", "user.email", "test@codedoctor.io"])

    commits = []

    # Commit 1: Initial auth
    auth_file = repo_dir / "auth.js"
    auth_file.write_text("function authenticateV1(user, pass) { return checkV1(user, pass); }\n", encoding="utf-8")
    run_git_cmd(repo_dir, ["add", "auth.js"])
    run_git_cmd(repo_dir, ["commit", "-m", "feat: initial authentication implementation v1"])
    c1 = run_git_cmd(repo_dir, ["rev-parse", "HEAD"])
    commits.append(c1)

    # Commit 2: Token session logic
    session_file = repo_dir / "session.js"
    session_file.write_text("function createSession(userId) { return jwtSign(userId); }\n", encoding="utf-8")
    run_git_cmd(repo_dir, ["add", "session.js"])
    run_git_cmd(repo_dir, ["commit", "-m", "feat: add JWT token session management"])
    c2 = run_git_cmd(repo_dir, ["rev-parse", "HEAD"])
    commits.append(c2)

    # Commit 3: Branch feature/oauth
    run_git_cmd(repo_dir, ["checkout", "-b", "feature/oauth"])
    oauth_file = repo_dir / "oauth.js"
    oauth_file.write_text("function handleOAuthCallback(code) { return exchangeOAuthCode(code); }\n", encoding="utf-8")
    run_git_cmd(repo_dir, ["add", "oauth.js"])
    run_git_cmd(repo_dir, ["commit", "-m", "feat: support OAuth2 callback workflow"])
    c3 = run_git_cmd(repo_dir, ["rev-parse", "HEAD"])
    commits.append(c3)

    # Commit 4: Back on main, update auth to v2
    run_git_cmd(repo_dir, ["checkout", "main"])
    auth_file.write_text("function authenticateV2(user, pass) { return checkV2(user, pass, true); }\n", encoding="utf-8")
    run_git_cmd(repo_dir, ["add", "auth.js"])
    run_git_cmd(repo_dir, ["commit", "-m", "refactor: update authentication to v2 with multi-factor"])
    c4 = run_git_cmd(repo_dir, ["rev-parse", "HEAD"])
    commits.append(c4)

    # Commit 5: Merge branch feature/oauth into main
    run_git_cmd(repo_dir, ["merge", "--no-ff", "feature/oauth", "-m", "merge: integrate OAuth feature branch"])
    c5 = run_git_cmd(repo_dir, ["rev-parse", "HEAD"])
    commits.append(c5)

    return repo_dir, commits


def test_plan_t9_zero_cross_commit_leakage(git_test_repo):
    repo_dir, commits = git_test_repo

    index = VersionedIndex()
    for c in commits:
        index.index_git_commit(repo_dir, c)

    queries = ["authenticate", "session", "oauth", "callback", "jwt", "user", "v1", "v2", "multi-factor"]

    # (a) At least 50 randomised trials, assert 0 leaks
    trials = 50
    leaks = 0
    for _ in range(trials):
        commit_choice = random.choice(commits)
        query_choice = random.choice(queries)

        results = index.retrieve(query_choice, commit=commit_choice, limit=10)
        no_leak = index.verify_no_cross_commit_leakage(results, commit_choice)
        if not no_leak:
            leaks += 1

    assert leaks == 0, f"Cross-commit leakage detected! Total leaks: {leaks}"
    print(f"\nPASS  {trials}/{trials} trials, 0 leaks")

    # (b) 5/5 'when did X change' fixtures -> correct commit in top 3
    fixtures = [
        ("initial authentication", commits[0]),
        ("session management jwt", commits[1]),
        ("OAuth2 callback", commits[2]),
        ("multi-factor v2", commits[3]),
        ("integrate OAuth feature", commits[4]),
    ]

    correct_in_top_3 = 0
    for query, expected_sha in fixtures:
        ranked = index.find_when_changed(query, repo_path=repo_dir, limit=3)
        top_shas = [r["commit"] for r in ranked]
        if expected_sha in top_shas:
            correct_in_top_3 += 1

    assert correct_in_top_3 == 5, f"Expected 5/5 correct commits in top 3, got {correct_in_top_3}/5"
    print("PASS  5/5 'when did X change' fixtures -> correct commit in top 3")

    # (c) Two near-identical file versions ordered correctly by change evidence
    # Commit 3 (v2 refactor) has the diff hunk for multi-factor
    diff_results = index.find_when_changed("multi-factor", repo_path=repo_dir, file_path="auth.js")
    assert diff_results
    assert diff_results[0]["commit"] == commits[3]
    print("PASS  two near-identical file versions ordered correctly by change evidence")

    # (d) Querying unindexed commit raises KeyError (no silent fallback to HEAD)
    with pytest.raises(KeyError, match="not indexed"):
        index.retrieve("auth", commit="unindexed_commit_999")
