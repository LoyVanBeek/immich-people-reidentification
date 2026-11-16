#!/usr/bin/env python3
"""Validate ground truth fixture photos exist and meet basic requirements."""

from pathlib import Path
import sys

PHOTOS_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "photos"

REQUIRED_PHOTOS = [
    # Alice (anchor = alice-1)
    "reid-gt-alice-1.jpg",
    "reid-gt-alice-2.jpg",
    "reid-gt-alice-3.jpg",
    # Bob (anchor = bob-1)
    "reid-gt-bob-1.jpg",
    "reid-gt-bob-2.jpg",
    "reid-gt-bob-3.jpg",
    # Carol (anchor = carol-1)
    "reid-gt-carol-1.jpg",
    "reid-gt-carol-2.jpg",
    "reid-gt-carol-3.jpg",
    # Negative pairs
    "reid-gt-neg-1.jpg",
    "reid-gt-neg-2.jpg",
]

OPTIONAL_PHOTOS = [
    "reid-gt-alice-4.jpg",
    "reid-gt-alice-5.jpg",
    "reid-gt-bob-4.jpg",
    "reid-gt-bob-5.jpg",
    "reid-gt-carol-4.jpg",
    "reid-gt-carol-5.jpg",
    "reid-gt-neg-3.jpg",
]


def check_jpeg(path: Path) -> tuple[bool, str]:
    """Basic JPEG validation: file exists, non-empty, starts with JPEG magic bytes."""
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        with path.open("rb") as f:
            header = f.read(2)
        if header != b"\xff\xd8":
            return False, "not a JPEG (missing FF D8 magic bytes)"
    except Exception as e:
        return False, f"read error: {e}"
    return True, "ok"


def main() -> int:
    print(f"Checking fixtures in: {PHOTOS_DIR}")
    print()

    all_ok = True

    # Required photos
    print("=== Required photos ===")
    for name in REQUIRED_PHOTOS:
        path = PHOTOS_DIR / name
        ok, msg = check_jpeg(path)
        status = "✅" if ok else "❌"
        print(f"  {status} {name}: {msg}")
        if not ok:
            all_ok = False

    # Optional photos
    print()
    print("=== Optional photos (extra variation) ===")
    for name in OPTIONAL_PHOTOS:
        path = PHOTOS_DIR / name
        ok, msg = check_jpeg(path)
        status = "✅" if ok else "⭕"
        print(f"  {status} {name}: {msg}")

    # Summary
    print()
    if all_ok:
        print("✅ All required fixtures present and valid")
        return 0
    else:
        print("❌ Missing or invalid required fixtures")
        print()
        print("Add the missing photos to:", PHOTOS_DIR)
        print("See .scratch/person-reid-v1/issues/16-add-ground-truth-fixtures.md for requirements")
        return 1


if __name__ == "__main__":
    sys.exit(main())