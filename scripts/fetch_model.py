"""Download public weights only, verifying the pinned artifact before replacing it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "models/generator.json").read_text())


def main(directory=None):
    if directory is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--directory", type=Path, default=ROOT / "models")
        directory = ap.parse_args().directory
    fetch(directory, SPEC)
    for item in json.loads((ROOT / "models/reranker.json").read_text()):
        fetch(directory / "reranker", item)


def fetch(directory, spec):
    dst = directory / spec["filename"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        with dst.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == spec["sha256"]:
                print(dst)
                return
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as f:
        temp = Path(f.name)
    try:
        h = hashlib.sha256()
        with (
            urllib.request.urlopen(
                spec["url"] + "?download=true", timeout=120
            ) as response,
            temp.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                h.update(chunk)
        if temp.stat().st_size != spec["bytes"] or h.hexdigest() != spec["sha256"]:
            raise ValueError("Размер или SHA-256 файла весов не совпадает с манифестом.")
        temp.chmod(0o644)
        os.replace(temp, dst)
        print(dst)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
