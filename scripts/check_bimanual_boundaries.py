"""Exercise the OFFICIAL Bimanual validator with a real sample and corrupted inputs (CPU only)."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, default=ROOT / "repos/cosmos-framework")
    p.add_argument(
        "--output", type=Path, default=ROOT / "reports/bimanual-boundaries.json"
    )
    args = p.parse_args()
    sys.path.insert(0, str(args.framework))
    from cosmos_framework.inference.bimanual import load_bimanual_action_contract

    snapshot = ROOT / "storage/models/Cosmos3-Nano-Sim-Bimanual"
    contract = load_bimanual_action_contract(snapshot)
    if contract is None:
        raise RuntimeError("The checkpoint did not expose the required action contract")
    real = json.loads(
        (
            snapshot
            / "assets/sim_bimanual/agibot/actions/agibot_001_normalized_unifiedv1.json"
        ).read_text()
    )
    count = len(real["action"])
    contract.validate(real, action_count=count, domain_name="agibotworld")
    results = [
        {
            "case": "official_59d_baseline",
            "expected": "accept",
            "actual": "accept",
            "passed": True,
        }
    ]
    cases = []

    def case(name, change):
        value = copy.deepcopy(real)
        change(value)
        cases.append((name, value))

    case("wrong_action_width_58", lambda v: v["action"][0].pop())
    case("missing_action_row", lambda v: v["action"].pop())
    case("nan_value", lambda v: v["action"][0].__setitem__(0, float("nan")))
    case("infinite_value", lambda v: v["action"][0].__setitem__(0, float("inf")))
    case("boolean_value", lambda v: v["action"][0].__setitem__(0, True))
    case(
        "wrong_action_space",
        lambda v: v.__setitem__("action_space", "raw_joint_positions"),
    )
    case(
        "unknown_domain",
        lambda v: v.__setitem__("domain_names", ["unsupported_robot"] * count),
    )
    case(
        "domain_count_mismatch",
        lambda v: v.__setitem__("domain_names", ["agibotworld"]),
    )

    def invalid_camera(v):
        v["domain_names"] = ["camera_pose"] * count
        v["action"][0][33] = 0.1

    case("camera_row_with_gripper_value", invalid_camera)
    for name, payload in cases:
        try:
            contract.validate(payload, action_count=count, domain_name="agibotworld")
            results.append(
                {
                    "case": name,
                    "expected": "reject",
                    "actual": "accept",
                    "passed": False,
                }
            )
        except ValueError as exc:
            results.append(
                {
                    "case": name,
                    "expected": "reject",
                    "actual": "reject",
                    "passed": True,
                    "reason": str(exc),
                }
            )
    report = {
        "scope": "CPU input-contract validation only; no model generation or quality evaluation.",
        "validator": "cosmos_framework.inference.bimanual.BimanualActionContract.validate",
        "passed": all(r["passed"] for r in results),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
