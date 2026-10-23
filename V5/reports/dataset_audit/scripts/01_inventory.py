"""
Stage 1 - read-only file inventory of every dataset folder.

For each dataset root:
  - file count, total bytes, extension histogram
  - archives / docs / scripts / label-like files
  - suspiciously small files (< 1 KiB)
  - duplicate detection: size + SHA-1 of first and last 64 KiB,
    confirmed with a full SHA-1 only when two files collide

Usage:  py -I 01_inventory.py <datasets_root> <out_json>
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict


# ----------------------------------------------------------------------
# Dataset roots (folder name -> working identity, verified later)
# ----------------------------------------------------------------------

DATASET_DIRS = [
    "CountFi",
    "Dataset",
    "Wiman",
    "WiCount",
    "IIT-BHU",
    "CSI HAR",
    "Haron98_WIFI-CSI-Dataset",
]

ARCHIVE_EXT = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz"}
DOC_EXT = {".md", ".txt", ".pdf", ".rst", ".html", ".docx"}
SCRIPT_EXT = {".py", ".m", ".ipynb", ".sh", ".ps1", ".r"}
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

SMALL_BYTES = 1024
EDGE_BYTES = 64 * 1024


# ----------------------------------------------------------------------
# Hash helpers
# ----------------------------------------------------------------------

def edge_hash(path, size):
    """SHA-1 of the first and last 64 KiB plus the size (cheap fingerprint)."""
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(EDGE_BYTES))
        if size > EDGE_BYTES:
            f.seek(max(size - EDGE_BYTES, EDGE_BYTES))
            h.update(f.read(EDGE_BYTES))
    return h.hexdigest()


def full_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Per-dataset walk
# ----------------------------------------------------------------------

def audit_dir(root, name):
    base = os.path.join(root, name)

    ext_counter = Counter()
    total_bytes = 0
    n_files = 0
    n_dirs = 0
    small_files = []
    archives, docs, scripts, junk, label_like = [], [], [], [], []
    errors = []
    fingerprints = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(base):
        n_dirs += 1
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            try:
                size = os.path.getsize(p)
            except OSError as e:
                errors.append(f"{rel}: stat failed: {e}")
                continue

            n_files += 1
            total_bytes += size
            ext = os.path.splitext(fn)[1].lower() or "(none)"
            ext_counter[ext] += 1

            low = fn.lower()
            if low in JUNK_NAMES:
                junk.append(rel)
            if ext in ARCHIVE_EXT:
                archives.append(rel)
            if ext in DOC_EXT or low.startswith(("readme", "license")):
                docs.append(rel)
            if ext in SCRIPT_EXT:
                scripts.append(rel)
            if "label" in low or "annotation" in low or "groundtruth" in low:
                label_like.append(rel)
            if size < SMALL_BYTES and low not in JUNK_NAMES:
                small_files.append({"path": rel, "bytes": size})

            try:
                fingerprints[edge_hash(p, size)].append(rel)
            except OSError as e:
                errors.append(f"{rel}: read failed: {e}")

    # Confirm candidate duplicates with a full hash
    duplicates = []
    for fp, paths in fingerprints.items():
        if len(paths) < 2:
            continue
        by_full = defaultdict(list)
        for rel in paths:
            try:
                by_full[full_hash(os.path.join(root, rel))].append(rel)
            except OSError as e:
                errors.append(f"{rel}: full hash failed: {e}")
        for h, group in by_full.items():
            if len(group) > 1:
                duplicates.append({"sha1": h, "files": group})

    return {
        "dataset_dir": name,
        "path": base.replace("\\", "/"),
        "n_files": n_files,
        "n_dirs": n_dirs,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 2**30, 3),
        "extensions": dict(ext_counter.most_common()),
        "archives": archives,
        "docs": docs,
        "scripts": scripts,
        "label_like_files": label_like[:50],
        "n_label_like_files": len(label_like),
        "junk_files": junk,
        "small_files": small_files,
        "exact_duplicate_groups": duplicates,
        "fingerprints": {fp: p for fp, p in fingerprints.items()},
        "errors": errors,
    }


# ----------------------------------------------------------------------
# Cross-dataset duplicate check (same fingerprint in two datasets)
# ----------------------------------------------------------------------

def cross_dupes(results):
    owner = defaultdict(set)
    for r in results:
        for fp in r["fingerprints"]:
            owner[fp].add(r["dataset_dir"])
    return [{"fingerprint": fp, "datasets": sorted(ds)}
            for fp, ds in owner.items() if len(ds) > 1]


def main():
    root, out = sys.argv[1], sys.argv[2]
    results = []
    for name in DATASET_DIRS:
        print("walking", name, flush=True)
        results.append(audit_dir(root, name))

    cross = cross_dupes(results)
    for r in results:
        del r["fingerprints"]

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"datasets": results, "cross_dataset_duplicates": cross},
                  f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
