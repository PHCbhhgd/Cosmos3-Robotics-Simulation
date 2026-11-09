"""Record CPU metadata, checkpoint completeness and shared-GPU availability; no CUDA kernels."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def query(fields, kind="gpu"):
    result = subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(result))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reports/preflight.json")
    args = p.parse_args()
    versions = json.loads((ROOT / "configs/versions.json").read_text())
    framework = ROOT / "repos/cosmos-framework"
    commit = subprocess.check_output(
        ["git", "-C", str(framework), "rev-parse", "HEAD"], text=True
    ).strip()
    models = {}
    for name, model in versions["models"].items():
        snapshot = ROOT / "storage/models" / model["repo_id"].split("/")[-1]
        metadata = json.loads((snapshot / "checkpoint.json").read_text())[
            "cosmos3_nano_sim_bimanual"
        ]
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        paths = sorted(set(index["weight_map"].values()))
        missing = [s for s in paths if not (snapshot / s).is_file()]
        models[name] = {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "attention": metadata["attention_mode"],
            "four_step_schedule_matches": metadata["fixed_step_sampler_config"][
                "t_list"
            ]
            == [1.0, 0.9375, 0.8333333333333334, 0.625],
            "transformer_bytes_from_index": index.get("metadata", {}).get("total_size"),
            "transformer_shards_required": len(paths),
            "transformer_shards_present": len(paths) - len(missing),
            "missing_transformer_shards": missing,
            "inference_attempted": False,
            "generation_passed": False,
        }
    processes = query("gpu_uuid,pid", "compute-apps")
    devices = []
    for index, uuid, name, capability, used, total, util in query(
        "index,uuid,name,compute_cap,memory.used,memory.total,utilization.gpu"
    ):
        count = sum(r[0] == uuid for r in processes)
        devices.append(
            {
                "index": int(index),
                "name": name,
                "compute_capability": capability,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_percent": int(util),
                "compute_process_count": count,
                "idle_at_observation": count == 0
                and int(used) < 1024
                and int(util) <= 5,
            }
        )
    report = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "framework_commit": commit,
        "framework_commit_matches_pin": commit == versions["source"]["commit"],
        "models": models,
        "gpus": devices,
        "status": "prepared_pending_allocated_gpu",
        "limitations": [
            "No inference was launched on GPUs already running compute jobs.",
            "RTX 4090/sm89 native three_way kernel compatibility is untested; the Sim-Auto gate is not evidence for these models.",
            "Each BF16 transformer alone is about 30.3 GB; a single 24 GiB GPU cannot hold all its unsharded weights.",
            "Idle observation is not a resource reservation. Obtain an allocation before launching.",
            "No generated video, physics quality, success rate, latency or stability duration is established.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
