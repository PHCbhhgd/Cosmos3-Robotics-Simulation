"""Create labeled previews of official inputs. These are NOT local model outputs."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"


def tile(image, label, width=400, height=225):
    image = ImageOps.contain(image.convert("RGB"), (width, height))
    result = Image.new("RGB", (width, height + 34), "#131d2a")
    result.paste(image, ((width - image.width) // 2, 34 + (height - image.height) // 2))
    ImageDraw.Draw(result).text((10, 11), label, fill="white")
    return result


def board(tiles, columns, filename):
    width, height = tiles[0].size
    image = Image.new(
        "RGB",
        (width * columns, height * ((len(tiles) + columns - 1) // columns) + 44),
        "#0b111c",
    )
    ImageDraw.Draw(image).text(
        (12, 15),
        "OFFICIAL INPUT DATA / NOT MODEL OUTPUT / NVIDIA Cosmos3",
        fill="#f1cf73",
    )
    for i, item in enumerate(tiles):
        image.paste(item, ((i % columns) * width, 44 + (i // columns) * height))
    image.save(MEDIA / filename, quality=90)


def frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot decode {path.name} at frame {index}")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main():
    MEDIA.mkdir(exist_ok=True)
    snap = ROOT / "storage/models/Cosmos3-Nano-Sim-Bimanual"
    photos = sorted((snap / "assets/sim_bimanual").glob("*/images/*.png"))
    board(
        [tile(Image.open(p), p.stem.replace("_initial", "")) for p in photos],
        3,
        "bimanual-inputs.jpg",
    )
    controls = (
        ROOT / "storage/models/Cosmos3-Nano-Sim-Transfer/assets/sample_data/example2"
    )
    board(
        [
            tile(
                frame(controls / f"example2{suffix}.mp4", 60),
                f"Official example2 | {label} | t=2s",
            )
            for suffix, label in [
                ("", "RGB"),
                ("_depth", "depth"),
                ("_segmentation", "segmentation"),
                ("_edge", "edge (downloaded example)"),
                ("_blur", "blur (downloaded example)"),
            ]
        ],
        3,
        "transfer-inputs.jpg",
    )
    gt = snap / "assets/training/agibot/gt/agibot_001_gt.mp4"
    board(
        [
            tile(frame(gt, i), f"Official AgiBot001 RGB | t={i/30:.0f}s")
            for i in [0, 60, 120]
        ],
        3,
        "robot-input.jpg",
    )
    panels = []
    for filename in [
        "agibot_001_normalized_unifiedv1.json",
        "agibot_mixed_007_normalized_mixed_action.json",
    ]:
        payload = json.loads(
            (snap / "assets/sim_bimanual/agibot/actions" / filename).read_text()
        )
        a = np.asarray(payload["action"], dtype=np.float32).T
        # Fixed display clipping is visualization only, never saved back to actions.
        scaled = np.clip(a, -3, 3) / 3
        heat = np.zeros((*scaled.shape, 3), dtype=np.uint8)
        heat[..., 0] = (255 * np.clip(scaled, 0, 1)).astype(np.uint8)
        heat[..., 2] = (255 * np.clip(-scaled, 0, 1)).astype(np.uint8)
        heat[..., 1] = (70 * (1 - np.abs(scaled))).astype(np.uint8)
        heatmap = Image.fromarray(heat).resize((900, 236), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (1120, 360), "#131d2a")
        panel.paste(heatmap, (195, 48))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (12, 12),
            filename + " | normalized values clipped to [-3,3] for display only",
            fill="white",
        )
        for label, row in [
            ("ego pose", 0),
            ("right wrist", 9),
            ("right fingertips", 18),
            ("right gripper", 33),
            ("left wrist", 34),
            ("left fingertips", 43),
            ("left gripper", 58),
        ]:
            y = 48 + row * 4
            label_y = y - 10 if row == 33 else y + 6 if row == 34 else y
            draw.text((10, label_y), label, fill="white")
            draw.line((125, label_y + 5, 190, y + 2), fill="#aaaaaa")
            draw.line((195, y, 1094, y), fill="#aaaaaa")
        names = payload.get("domain_names", ["agibotworld"] * a.shape[1])
        if isinstance(names, str):
            names = [names] * a.shape[1]
        colors = np.array(
            [[245, 179, 66] if n == "camera_pose" else [72, 182, 144] for n in names],
            dtype=np.uint8,
        )[None]
        panel.paste(
            Image.fromarray(colors).resize((900, 12), Image.Resampling.NEAREST),
            (195, 292),
        )
        for time in (0, 10, 20, 30):
            x = 195 + round(time / 30 * 899)
            draw.line((x, 305, x, 310), fill="white")
            draw.text((x - (20 if time == 30 else 0), 313), f"{time}s", fill="white")
        draw.text(
            (195, 339), "Domain routing: green = robot; yellow = camera", fill="white"
        )
        panels.append(panel)
    board(panels, 1, "action-trajectory.jpg")


if __name__ == "__main__":
    main()
