#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate xplace_work/contest.json for a design")
    parser.add_argument("design_name", help="design name passed from run_xplace.sh")
    parser.add_argument(
        "--template",
        default=Path("xplace_work/contest.json"),
        type=Path,
        help="template contest.json to update",
    )
    parser.add_argument(
        "--output",
        default=Path("xplace_work/contest.json"),
        type=Path,
        help="output contest.json path",
    )
    args = parser.parse_args()

    with args.template.open("r", encoding="utf-8") as handle:
        contest = json.load(handle)

    contest["design_name"] = args.design_name
    contest["def"] = f"/workspace/xplace_work/{args.design_name}_unplaced.def"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(contest, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
