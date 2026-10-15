from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent

def equal(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            equal(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, path + "[" + str(i) + "]")
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), path
    else:
        assert type(actual) is type(expected) and actual == expected, path

def main():
    files = sorted((ROOT / "control_results").glob("*.json"))
    assert files, "control results missing"
    for source in files:
        target = ROOT / "lab_results" / source.name
        equal(json.loads(target.read_text(encoding="utf-8")),
              json.loads(source.read_text(encoding="utf-8")), source.name)
    print("Verified", len(files), "control reports")

if __name__ == "__main__":
    main()
