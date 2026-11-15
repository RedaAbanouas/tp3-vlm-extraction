"""Compare extraction JSON files against out/ground_truth.json.

Usage: python3 evaluate_json.py [prediction.json ...]

With no file arguments, evaluates every out/*.json except ground_truth.json and
latency_api.json. Scores are micro-averaged over expected scalar fields. Item
arrays are matched by their stable `code`; labels and model metadata are not
scored. Missing and extra item codes are reported as structural errors.
"""

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "out"
ITEM_FIELDS = {
    "assets": (
        "label",
        "gross",
        "depreciation_and_provisions",
        "net_current",
        "net_previous",
    ),
    "liabilities_and_equity": ("label", "current", "previous"),
}
TOTAL_SECTIONS = ("assets", "liabilities_and_equity")
SUMMARY_PATH = OUT / "model_benchmark.json"

# Parameter counts are in billions. For dense models, all parameters are active.
MODEL_PARAMETERS = {
    "qwen2.5vl:3b": (3, 3),
    "qwen2.5vl:7b": (7, 7),
    "qwen2.5vl:32b": (32, 32),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": (30, 3),
    "nvidia/nemotron-3.5-super-vl:free": (120, 12),
}


def normalize(value):
    """Normalize numeric strings while preserving null/empty distinctions."""
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    # French thousands separators and decimal commas; remove normal/narrow NBSP.
    compact = re.sub(r"[\s\u00a0\u202f]", "", text).replace(",", ".")
    try:
        return Decimal(compact)
    except InvalidOperation:
        return text.casefold()


def evaluate(truth, prediction):
    correct = total = 0
    errors = []

    def check(path, expected, actual):
        nonlocal correct, total
        total += 1
        if normalize(expected) == normalize(actual):
            correct += 1
        else:
            errors.append((path, "wrong value", expected, actual))

    # Fixed scalar schema paths.
    for key in ("document_type",):
        check(key, truth.get(key), prediction.get(key))
    for key in ("start_date", "end_date"):
        check(
            f"period.{key}",
            truth.get("period", {}).get(key),
            prediction.get("period", {}).get(key),
        )

    # Item fields: rows are associated by stable code, not array position.
    structure_errors = 0
    for section, fields in ITEM_FIELDS.items():
        expected_items = truth.get(section, {}).get("items", []) or []
        actual_items = prediction.get(section, {}).get("items", []) or []
        expected_by_code = {
            str(row.get("code")): row
            for row in expected_items
            if row.get("code") is not None
        }
        actual_by_code = {
            str(row.get("code")): row
            for row in actual_items
            if row.get("code") is not None
        }
        for code in expected_by_code.keys() - actual_by_code.keys():
            errors.append(
                (
                    f"{section}.items[{code}]",
                    "missing item",
                    expected_by_code[code],
                    None,
                )
            )
            structure_errors += 1
        for code in actual_by_code.keys() - expected_by_code.keys():
            errors.append(
                (f"{section}.items[{code}]", "extra item", None, actual_by_code[code])
            )
            structure_errors += 1
        for code in expected_by_code.keys() & actual_by_code.keys():
            for field in fields:
                check(
                    f"{section}.items[{code}].{field}",
                    expected_by_code[code].get(field),
                    actual_by_code[code].get(field),
                )

    # Totals include all expected schema keys; missing keys count as errors.
    for section in TOTAL_SECTIONS:
        expected_totals = truth.get(section, {}).get("totals", {}) or {}
        actual_totals = prediction.get(section, {}).get("totals", {}) or {}
        for key, expected in expected_totals.items():
            check(f"{section}.totals.{key}", expected, actual_totals.get(key))

    accuracy = 100 * correct / total if total else 0.0
    return correct, total, accuracy, errors, structure_errors


def main():
    truth_path = OUT / "ground_truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        files = [Path(arg) for arg in sys.argv[1:]]
    else:
        files = sorted(
            p
            for p in OUT.glob("*.json")
            if p.name
            not in {
                "ground_truth.json",
                "latency_api.json",
                "latency_local.json",
                SUMMARY_PATH.name,
            }
        )

    local_latencies = json.loads((OUT / "latency_local.json").read_text(encoding="utf-8")) if (OUT / "latency_local.json").exists() else {}
    api_latencies = json.loads((OUT / "latency_api.json").read_text(encoding="utf-8")) if (OUT / "latency_api.json").exists() else {}
    benchmark = {"models": []}

    for path in files:
        prediction = json.loads(path.read_text(encoding="utf-8"))
        _, _, accuracy, _, _ = evaluate(truth, prediction)
        model_name = prediction.get("model", path.stem)
        print(f"{model_name}: {accuracy:.2f}%")

        if model_name in local_latencies:
            provider, latency = "local", local_latencies[model_name]
        elif model_name in api_latencies:
            provider, latency = "api", api_latencies[model_name]
        else:
            provider, latency = None, None
        params = MODEL_PARAMETERS.get(model_name, (None, None))
        benchmark["models"].append(
            {
                "name": model_name,
                "provider": provider,
                "latency_seconds": latency,
                "parameters_billions": params[0],
                "active_parameters_billions": params[1],
                "accuracy_percent": round(accuracy, 2),
            }
        )

    SUMMARY_PATH.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
