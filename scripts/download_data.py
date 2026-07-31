"""Download ViRHE4QA after explicit license acknowledgement."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

VIRHE4QA_URL = (
    "https://github.com/DoPhamPhucTinh/R2GQA/raw/refs/heads/main/ViRHE4QA.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/ViRHE4QA.zip"))
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that you reviewed and accept the upstream research-use terms.",
    )
    args = parser.parse_args()
    if not args.accept_license:
        raise SystemExit(
            "Download cancelled. Review data/README.md and rerun with "
            "--accept-license if the upstream terms are acceptable."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(VIRHE4QA_URL, args.output)
    print(f"Downloaded local, Git-ignored archive to {args.output}")


if __name__ == "__main__":
    main()
