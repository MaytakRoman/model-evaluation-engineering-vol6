from pathlib import Path
import importlib
import json
import platform

ROOT = Path(__file__).resolve().parent
MODULES = ['protocol', 'benchmark', 'statistics_eval', 'judge', 'ratings', 'agent_eval', 'safety_eval', 'monitor_eval', 'acceptance_eval']

def main():
    output = ROOT / "lab_results"
    output.mkdir(exist_ok=True)
    reports = {}
    for name in MODULES:
        module = importlib.import_module("programs." + name)
        if name in {"protocol", "agent_eval"}:
            reports[name] = module.demo(output_dir=output)
        else:
            reports[name] = module.demo()
    from programs.acceptance_eval import evaluate
    records = [json.loads(line) for line in (
        ROOT / "data/acceptance_cases.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    config = json.loads((ROOT / "configs/acceptance.json").read_text(encoding="utf-8"))
    reports["acceptance_from_file"] = evaluate(records, config)
    for name, report in reports.items():
        (output / (name + ".json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "environment.json").write_text(
        json.dumps({"python": platform.python_version(),
                    "implementation": platform.python_implementation()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Completed", len(reports), "reports in lab_results")

if __name__ == "__main__":
    main()
