from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_visualizations(source: Path, dest: Path):
    if not source.exists():
        raise FileNotFoundError(f"Folder segmentasi tidak ditemukan: {source}")
    patterns = ["*_vis.jpg", "*_vis.jpeg", "*_vis.png", "*_vis.webp"]
    files = []
    for pattern in patterns:
        files.extend(source.rglob(pattern))
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(files):
        rel = src.relative_to(source)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    print(f"Copied {len(files)} visualization files to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Copy SAM3 visualization files only, not .npy masks.")
    parser.add_argument("--source", default="../output_segmentasi")
    parser.add_argument("--dest", default="assets/output_segmentasi")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    copy_visualizations((project_dir / args.source).resolve(), (project_dir / args.dest).resolve())


if __name__ == "__main__":
    main()
