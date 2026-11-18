"""Download the pinned, multilingual pretrained model without global caches."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def valid(path, details):
    if not path.is_file() or path.stat().st_size != details["size"]:
        return False
    payload = path.read_bytes()
    if "sha256" in details:
        return hashlib.sha256(payload).hexdigest() == details["sha256"]
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() == details["git_blob"]


def main():
    manifest = json.loads((ROOT / "config/asr_model.json").read_text(encoding="utf-8"))
    folder = ROOT / manifest["local_directory"]
    folder.mkdir(parents=True, exist_ok=True)
    for name, details in manifest["files"].items():
        path = folder / name
        if not valid(path, details):
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}?download=true"
            request = urllib.request.Request(url, headers={"User-Agent": "ShowIntent-Speech"})
            temp = path.with_suffix(path.suffix + ".part")
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        if stream.tell() > details["size"]:
                            raise ValueError(f"unexpected download size: {name}")
                if not valid(temp, details):
                    raise ValueError(f"checksum mismatch: {name}")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
