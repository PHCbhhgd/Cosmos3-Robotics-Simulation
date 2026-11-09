"""Prepare edge/blur robot requests from OFFICIAL recorded RGB (not generated video)."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "storage/models/Cosmos3-Nano-Sim-Bimanual/assets/training/agibot/gt/agibot_001_gt.mp4",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    frames = 121
    rgb = (args.output / "robot_rgb_121.mp4").resolve()
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 30) > 0.01:
        raise ValueError("Expected the official 30 FPS source")
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        str(rgb), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("No MP4 encoder available")
    try:
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Source has fewer than 121 frames")
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    sample = json.loads(
        (
            ROOT
            / "storage/models/Cosmos3-Nano-Sim-Bimanual/examples/inference/sim_bimanual_unifiedv1_hf.jsonl"
        )
        .read_text()
        .splitlines()[0]
    )
    defaults = json.loads(
        (
            ROOT
            / "repos/cosmos-framework/.agents/skills/cosmos3-nano-sim-transfer/inference/assets/sample_args.json"
        ).read_text()
    )
    for modality, setting in [
        ("edge", "preset_edge_threshold"),
        ("blur", "preset_blur_strength"),
    ]:
        request = {
            **defaults,
            "name": f"robot_agibot001_{modality}_121",
            "model_mode": "video2video",
            "vision_path": str(rgb),
            "prompt": sample["prompt"],
            "fps": 30,
            "num_frames": frames,
            "max_frames": frames,
            "num_video_frames_per_chunk": frames,
            "aspect_ratio": "16,9",
            "seed": 2025,
            modality: {setting: "medium"},
        }
        (args.output / f"{modality}.json").write_text(
            json.dumps(request, indent=2) + "\n"
        )
    with args.source.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "source": "official AgiBot 001 ground-truth RGB",
                "source_sha256": sha,
                "first_source_frame": 0,
                "selected_frames": frames,
                "fps": 30,
                "caption_source": sample["name"],
                "caption_edits": False,
                "control_preprocessing": "Native pipeline resizes RGB to 832x480 before computing edge/blur. No synthetic depth or segmentation.",
                "status": "prepared_only_not_inferred",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
