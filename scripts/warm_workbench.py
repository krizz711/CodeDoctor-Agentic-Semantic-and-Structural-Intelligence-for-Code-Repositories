"""Pre-encode the demo corpora so the workbench starts warm.

The interactive encoder is a 0.6B transformer on CPU, which costs a few
seconds per chunk.  That is fine once and unacceptable per page load, so the
vectors are computed ahead of time into ``build/cache.db`` -- the same cache
the incremental indexer and the live server read, keyed on the encoder input.

    python scripts/warm_workbench.py --target self --versions 4
    python scripts/warm_workbench.py --target commander --versions 1

Re-running is cheap: anything already cached is skipped, so this is also how
you top the cache up after a commit.  Without it the workbench still answers
queries, lexically, and says so -- it never pretends a dense pass ran.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.indexing.cache import EmbeddingCache  # noqa: E402
from app.workbench import TARGETS, Workbench  # noqa: E402

CACHE_PATH = REPO_ROOT / "build" / "cache.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="self", choices=sorted(TARGETS))
    parser.add_argument("--versions", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args(argv)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    bench = Workbench(cache=EmbeddingCache(str(args.cache)))

    target = TARGETS[args.target]
    available, reason = bench._availability(target)
    if not available:
        print(f"{args.target} is unavailable: {reason}")
        return 1

    print(f"warming {target.label} ({target.language}), {args.versions} version(s)")
    print(f"cache: {args.cache}")
    started = time.time()
    bench.start_build(args.target, versions=args.versions, dense=True)

    last = ""
    while True:
        state = bench.state()
        line = (
            f"[{state['phase']:>8}] v{state['versions_done']}/{state['versions_total']} "
            f"encoded {state['encoded']}/{state['encode_total']} "
            f"reused {state['reused']} — {state['message']}"
        )
        if line != last:
            print(line, flush=True)
            last = line
        if state["phase"] in {"ready", "error"}:
            break
        time.sleep(2.0)

    state = bench.state()
    if state["phase"] == "error":
        print(f"failed: {state['error']}")
        return 1

    elapsed = time.time() - started
    print(f"done in {elapsed / 60:.1f} min — {state['chunks']} chunks across "
          f"{state['versions_done']} version(s), dense={state['dense']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
