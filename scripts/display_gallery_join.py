#!/usr/bin/env python3
"""Join two display-gallery documents into one before/after table.

    display_gallery_join.py before.md after.md > before-after.md

Both documents come from the same gallery source, so their sections and rows
line up; a row is matched by its item column, and an unmatched row is reported
rather than dropped.
"""

import sys
from pathlib import Path


def parse(path):
    """The document as [(section title, [(item, rendering)])]."""
    sections = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.startswith("| ") and not line.startswith("| --- ") and sections:
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            if cells[:2] != ["item", "rendering"]:
                sections[-1][1].append((cells[0], cells[1]))
    return sections


def main():
    before, after = (parse(path) for path in sys.argv[1:3])

    print("# hdf5-pure display gallery, before and after")
    print()
    print(
        "Every value the crate describes to a caller. *Before* is what a caller "
        "had without these `Display` impls, which for most of these types means "
        "the `Debug` record."
    )

    for (title, before_rows), (after_title, after_rows) in zip(before, after):
        assert title == after_title, f"{title} != {after_title}"
        print()
        print(f"## {title}")
        print()
        print("| item | before | after |")
        print("| --- | --- | --- |")

        before_by_item = dict(before_rows)
        for item, after_rendering in after_rows:
            before_rendering = before_by_item.pop(item, "*(row absent)*")
            print(f"| {item} | {before_rendering} | {after_rendering} |")
        for item in before_by_item:
            print(f"| {item} | {before_by_item[item]} | *(row absent)* |")


if __name__ == "__main__":
    main()
