"""Fully decode downloaded official input videos and record structure and hashes."""

import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def inspect(path, expected, source):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dimensions = [
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    ]
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if [frame.shape[1], frame.shape[0]] != dimensions:
            raise ValueError("Video changes dimensions")
        count += 1
    cap.release()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "video": path.name,
        "source": source,
        "frames": count,
        "declared_frames": declared,
        "expected_frames": expected,
        "fps": fps,
        "dimensions": dimensions,
        "last_frame_timestamp_s": (count - 1) / fps,
        "sha256": sha,
        "full_decode": count == declared == expected,
        "passed": count == declared == expected and abs(fps - 30) < 0.01,
    }


def main():
    base = ROOT / "storage/models/Cosmos3-Nano-Sim-Transfer/assets/sample_data/example2"
    rows = [
        inspect(p, 121, "official Transfer example2 input")
        for p in sorted(base.glob("*.mp4"))
    ]
    gt = (
        ROOT
        / "storage/models/Cosmos3-Nano-Sim-Bimanual/assets/training/agibot/gt/agibot_001_gt.mp4"
    )
    rows += [
        inspect(gt, 901, "official Bimanual recorded ground-truth RGB"),
        inspect(
            ROOT / "artifacts/robot-transfer/robot_rgb_121.mp4",
            121,
            "locally selected official RGB prefix; re-encoded with mp4v",
        ),
    ]
    report = {
        "scope": "INPUT integrity only. These are not model inference outputs.",
        "passed": len(rows) == 7 and all(r["passed"] for r in rows),
        "videos": rows,
    }
    (ROOT / "reports/input-videos.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
