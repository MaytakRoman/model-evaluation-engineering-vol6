from pathlib import Path
from hashlib import sha256

ROOT = Path(__file__).resolve().parent

def main():
    count = 0
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("checksum mismatch: " + name)
        count += 1
    print("Verified", count, "file checksums")

if __name__ == "__main__":
    main()
