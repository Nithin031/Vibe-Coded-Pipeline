"""Non-destructive repository snapshot: Git status, diffs, file inventory, V3 hashes.

Writes into V4/manifests/snapshot/:

    git_status_v3.txt        `git status` of the V3 repository (its own .git)
    git_diff_v3_stat.txt     `git diff --stat` of uncommitted V3 changes
    git_log_v3.txt           last commits of V3
    file_inventory.csv       every file under the project (except datasets/, archive/, caches)
    v3_hashes.json           SHA-256 of every V3 model, result, figure, metric and manifest

v3_hashes.json is the reference for the "V3 unchanged" test (tests/test_v3_frozen.py).
Nothing is modified, reset or committed.

Usage:  python V4/src/snapshot_repo.py            write the snapshot (refuses to overwrite hashes)
        python V4/src/snapshot_repo.py --check    compare V3 against the saved hashes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "V3" if (ROOT / "V3" / "src" / "deep.py").exists() else ROOT    # project or V3 Git repo
OUT = ROOT / "V4" / "manifests" / "snapshot"

# the local project snapshot (taken from V3's working tree) and the committed V3 'main'
# can differ, so each layout has its own reference file
HASH_FILE = "v3_hashes.json" if V3 != ROOT else "v3_hashes_repo_main.json"

# V3 folders whose content must never change
V3_FROZEN = ["models", "results", "manifests", "src", "docs"]
SKIP_PARTS = {"logs", "__pycache__", "V4"}
SKIP_DIRS = {"datasets", "archive", "cache", "__pycache__", ".git", "logs"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args, cwd=V3) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout + result.stderr


def v3_hashes() -> dict:
    """SHA-256 of every frozen V3 artifact, keyed by path relative to V3/."""
    hashes = {}
    for folder in V3_FROZEN:
        for path in sorted((V3 / folder).rglob("*")):
            if path.is_file() and not SKIP_PARTS.intersection(path.relative_to(V3).parts):
                hashes[path.relative_to(V3).as_posix()] = sha256(path)
    return hashes


def inventory():
    """(relative path, bytes, modified) for project files outside data / caches."""
    rows = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or SKIP_DIRS.intersection(rel.parts):
            continue
        stat = path.stat()
        rows.append((rel.as_posix(), stat.st_size, time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))))
    return rows


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def write_snapshot():
    OUT.mkdir(parents=True, exist_ok=True)
    hash_path = OUT / HASH_FILE
    if hash_path.exists():
        sys.exit(f"{hash_path} already exists; the reference snapshot is immutable (use --check)")

    if HASH_FILE != "v3_hashes.json":                 # repo layout: hashes only
        hashes = v3_hashes()
        hash_path.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "layout": "V3 git repo",
                                         "n_files": len(hashes), "sha256": hashes}, indent=2))
        print(f"repo-layout hashes written ({len(hashes)} files)")
        return
    (OUT / "git_status_v3.txt").write_text(git("status") + "\n--- short ---\n" + git("status", "--short"))
    (OUT / "git_diff_v3_stat.txt").write_text(git("diff", "--stat"))
    (OUT / "git_log_v3.txt").write_text(git("log", "--oneline", "-n", "20") + "\n" + git("remote", "-v"))

    with (OUT / "file_inventory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "modified"])
        writer.writerows(inventory())

    hashes = v3_hashes()
    hash_path.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"),
                                     "n_files": len(hashes), "sha256": hashes}, indent=2))
    print(f"snapshot written to {OUT}  ({len(hashes)} V3 files hashed)")


def check() -> list[str]:
    """Return a list of V3 files that changed, appeared or disappeared."""
    saved = json.loads((OUT / HASH_FILE).read_text())["sha256"]
    now = v3_hashes()
    problems = [f"changed: {k}" for k in saved if k in now and now[k] != saved[k]]
    problems += [f"missing: {k}" for k in saved if k not in now]
    problems += [f"new: {k}" for k in now if k not in saved]
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare V3 against the saved hashes")
    args = parser.parse_args()

    if args.check:
        problems = check()
        print("V3 unchanged" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)
    write_snapshot()


if __name__ == "__main__":
    main()
