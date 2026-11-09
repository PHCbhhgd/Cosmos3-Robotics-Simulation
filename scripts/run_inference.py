"""Launch the official native pipeline, recording commands and refusing busy GPUs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_command(
    model,
    framework,
    snapshot,
    input_path,
    output,
    gpus,
    vae=None,
    tokenizer=None,
    long=False,
):
    common = [
        "-m",
        "cosmos_framework.scripts.inference",
        "-i",
        str(input_path),
        "-o",
        str(output),
        "--checkpoint-path",
        str(snapshot),
        "--config-file",
        str(snapshot / "config.json"),
        "--credential-path",
        "",
        "--autoregressive=True",
        "--num-steps",
        "4",
        "--max-num-seqs",
        "1",
        "--no-use-torch-compile",
        "--no-diffusion-cache",
        "--prompt-upsampling=False",
    ]
    if len(gpus) > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(gpus)}",
            *common,
        ]
    else:
        cmd = [sys.executable, *common]
    cmd += [
        "--cp-size",
        "1",
        "--cfgp-size",
        "1",
        "--dp-shard-size",
        str(len(gpus)),
        "--dp-replicate-size",
        "1",
    ]
    overrides = []
    if model == "bimanual":
        cmd += [
            "--model-mode=FORWARD_DYNAMICS",
            "--resolution=480",
            "--image-size=480",
            "--seed",
            "2026",
            "--kv-cache-inference-size",
            "64",
            "--attention-sink-size",
            "8",
            "--no-guardrails",
            "--video-save-quality",
            "4",
        ]
    else:
        cmd += [
            "--defaults-file",
            str(
                framework
                / ".agents/skills/cosmos3-nano-sim-transfer/inference/assets/sample_args.json"
            ),
            "--seed",
            "2025",
            "--num-outputs",
            "1",
            "--guidance",
            "1",
            "--control-guidance",
            "1",
            "--kv-cache-inference-size",
            "30" if long else "0",
            "--attention-sink-size",
            "3" if long else "0",
            "--no-use-cuda-graphs",
            "--no-condition-stream-encode",
        ]
        overrides += [
            "model.config.teacher_forcing_frames_per_chunk=1",
            "model.config.kv_cache_dtype=null",
        ]
    if vae and tokenizer:
        overrides += [
            "model.config.tokenizer.bucket_name=",
            "model.config.tokenizer.object_store_credential_path_pretrained=",
            f"model.config.tokenizer.vae_path={vae}",
            "model.config.vlm_config.tokenizer.config_variant=hf",
            f"model.config.vlm_config.tokenizer.pretrained_model_name={tokenizer}",
            "model.config.vlm_config.pretrained_weights.enabled=False",
        ]
    if overrides:
        cmd += ["--experiment-overrides", *overrides]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["bimanual", "transfer"])
    p.add_argument("--framework", type=Path, default=ROOT / "repos/cosmos-framework")
    p.add_argument(
        "--gpus", required=True, help="Physical GPU indices already allocated to you"
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--vae", type=Path)
    p.add_argument("--text-tokenizer", type=Path)
    p.add_argument(
        "--long", action="store_true", help="Transfer example1 KV-30 / sink-3"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only. This does not validate GPU compatibility.",
    )
    args = p.parse_args()
    gpus = [s.strip() for s in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or not all(s.isdigit() for s in gpus):
        p.error("Select distinct physical GPU indices")
    framework, input_path, output = (
        args.framework.resolve(),
        args.input.resolve(),
        args.output.resolve(),
    )
    versions = json.loads((ROOT / "configs/versions.json").read_text())
    snapshot = (
        ROOT
        / "storage/models"
        / versions["models"][args.model]["repo_id"].split("/")[-1]
    )
    if args.model == "transfer" and not (args.vae and args.text_tokenizer):
        p.error(
            "Transfer requires --vae and --text-tokenizer (matched local resources)"
        )
    if bool(args.vae) != bool(args.text_tokenizer):
        p.error("Pass --vae and --text-tokenizer together")
    vae = args.vae.resolve() if args.vae else None
    tokenizer = args.text_tokenizer.resolve() if args.text_tokenizer else None
    cmd = build_command(
        args.model,
        framework,
        snapshot,
        input_path,
        output,
        gpus,
        vae,
        tokenizer,
        args.long,
    )
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    if output.exists():
        p.error(
            "Use a fresh output directory; previous results will not be overwritten"
        )
    if not input_path.is_file():
        p.error("Input request does not exist")
    if (
        subprocess.check_output(
            ["git", "-C", str(framework), "rev-parse", "HEAD"], text=True
        ).strip()
        != versions["source"]["commit"]
    ):
        p.error("Framework commit differs from configs/versions.json")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_gpus.py"), "--gpus", args.gpus],
        check=True,
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    missing = [
        f
        for f in sorted(set(index["weight_map"].values()))
        if not (snapshot / f).is_file()
    ]
    if missing:
        p.error("Missing transformer shards: " + ", ".join(missing))
    if vae and (not vae.is_file() or not tokenizer.is_dir()):
        p.error("VAE or tokenizer is not materialized")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(gpus),
        COSMOS_INTERNAL="0",
        COSMOS_TRAINING="0" if args.model == "bimanual" else "1",
        WANDB_MODE="disabled",
        PYTHONPATH=str(framework),
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if args.model == "bimanual":
        subprocess.run(
            [
                sys.executable,
                str(framework / "tools/check_bimanual_mixed_inputs.py"),
                "--checkpoint-path",
                str(snapshot),
                "--input",
                str(input_path),
            ],
            env=env,
            check=True,
        )
    # Recheck immediately before launch. This is not a resource scheduler or reservation.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_gpus.py"), "--gpus", args.gpus],
        check=True,
    )
    output.mkdir(parents=True)
    receipt = {
        "model": args.model,
        "model_revision": versions["models"][args.model]["revision"],
        "framework_commit": versions["source"]["commit"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "physical_gpus": gpus,
        "status": "running",
        "note": "Process return code alone is not output-integrity or physics validation.",
    }
    record = output / "launch.json"
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (output / "console.log").open("w") as log:
            result = subprocess.run(
                cmd, cwd=framework, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        receipt.update(
            returncode=result.returncode,
            status="process_exited_zero" if result.returncode == 0 else "failed",
        )
        return result.returncode
    except BaseException as exc:
        receipt.update(status="interrupted_or_launch_failed", error=type(exc).__name__)
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
