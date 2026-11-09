"""Prepare local requests from pinned official inputs, preserving routing and normalization."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def local_asset(url, snapshot):
    parsed = urlsplit(url)
    prefix = "/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/main/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or not parsed.path.startswith(prefix)
    ):
        raise ValueError("Unexpected official asset URL")
    path = (snapshot / unquote(parsed.path[len(prefix) :])).resolve()
    if not path.is_relative_to(snapshot.resolve()) or not path.is_file():
        raise ValueError("Missing asset or asset outside snapshot")
    return path


def bimanual(snapshot, output, sample, count=None):
    manifest = snapshot / "examples/inference/sim_bimanual_unifiedv1_hf.jsonl"
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    matches = [r for r in rows if r["name"] == sample]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous sample: {sample}")
    row = copy.deepcopy(matches[0])
    original = copy.deepcopy(row)
    payload = json.loads(local_asset(row["action_path"], snapshot).read_text())
    total = row["num_frames"] - 1
    count = total if count is None else count
    if type(count) is not int or count <= 0 or count % 4 or count > total:
        raise ValueError(
            f"Action count must be positive, divisible by 4, and <= {total}"
        )
    output.mkdir(parents=True, exist_ok=True)
    payload["action"] = payload["action"][:count]
    if isinstance(payload.get("domain_names"), list):
        payload["domain_names"] = payload["domain_names"][:count]
    action_file = (output / "actions.json").resolve()
    action_file.write_text(json.dumps(payload) + "\n")
    row.update(
        vision_path=str(local_asset(row["vision_path"], snapshot)),
        action_path=str(action_file),
        action_chunk_size=count,
        num_frames=count + 1,
    )
    if count != total:
        row["name"] += f"_prefix_{count}"
    request = output / "input.jsonl"
    request.write_text(json.dumps(row) + "\n")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_sample": original,
                "localization_only": count == total,
                "prefix_action_count": count,
                "original_action_count": total,
                "note": "Prefix variants are smoke tests. No actions are renormalized or synthesized.",
            },
            indent=2,
        )
        + "\n"
    )
    return request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["bimanual", "transfer"])
    p.add_argument("--framework", type=Path, default=ROOT / "repos/cosmos-framework")
    p.add_argument("--sample", default="agibot_001_bulk_seed2026")
    p.add_argument("--action-count", type=int)
    p.add_argument("--example", choices=["example1", "example2"], default="example2")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output directory to preserve previous requests")
    if args.model == "bimanual":
        print(
            bimanual(
                ROOT / "storage/models/Cosmos3-Nano-Sim-Bimanual",
                args.output,
                args.sample,
                args.action_count,
            )
        )
    else:
        script = (
            args.framework
            / ".agents/skills/cosmos3-nano-sim-transfer/scripts/prepare_hf_sample.py"
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--example",
                args.example,
                "--sample-dir",
                str(
                    ROOT
                    / "storage/models/Cosmos3-Nano-Sim-Transfer/assets/sample_data"
                    / args.example
                ),
                "--output",
                str(args.output),
            ],
            check=True,
        )
        print(args.output / "requests")


if __name__ == "__main__":
    main()
