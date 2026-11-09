import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / (name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module("prepare_inputs")
runner = module("run_inference")
audit = module("audit_public_repo")


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / "snapshot"
        (self.snapshot / "assets").mkdir(parents=True)
        (self.snapshot / "examples/inference").mkdir(parents=True)
        self.payload = {
            "action_space": "normalized_unified_v1",
            "action": [[i / 10] * 59 for i in range(16)],
            "domain_names": ["agibotworld"] * 8 + ["camera_pose"] * 8,
        }
        (self.snapshot / "assets/a.json").write_text(json.dumps(self.payload))
        (self.snapshot / "assets/i.png").write_bytes(b"image")
        self.row = {
            "name": "real-routing-shape",
            "num_frames": 17,
            "action_chunk_size": 16,
            "fps": 30,
            "prompt": "Keep the original caption.",
            "domain_name": "agibotworld",
            "vision_path": "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/main/assets/i.png",
            "action_path": "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/main/assets/a.json",
        }
        (
            self.snapshot / "examples/inference/sim_bimanual_unifiedv1_hf.jsonl"
        ).write_text(json.dumps(self.row) + "\n")

    def test_aligned_prefix_preserves_normalized_rows_and_routing(self):
        path = prepare.bimanual(
            self.snapshot, self.root / "short", self.row["name"], 12
        )
        row = json.loads(path.read_text())
        payload = json.loads(Path(row["action_path"]).read_text())
        self.assertEqual(payload["action"], self.payload["action"][:12])
        self.assertEqual(payload["domain_names"], self.payload["domain_names"][:12])
        self.assertEqual(row["num_frames"], 13)
        self.assertEqual(row["action_chunk_size"], 12)
        self.assertEqual(row["prompt"], self.row["prompt"])

    def test_unaligned_empty_and_oversize_trajectories_rejected(self):
        for count in [0, -4, 5, 20]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                prepare.bimanual(
                    self.snapshot, self.root / "bad", self.row["name"], count
                )
        self.assertFalse((self.root / "bad").exists())

    def test_asset_traversal_and_foreign_host_rejected(self):
        for url in [
            self.row["vision_path"].replace("huggingface.co", "example.org"),
            self.row["vision_path"].replace("assets/i.png", "../../outside"),
        ]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                prepare.local_asset(url, self.snapshot)


class LaunchTests(unittest.TestCase):
    def test_bimanual_uses_four_step_action_route(self):
        cmd = runner.build_command(
            "bimanual",
            Path("/framework"),
            Path("/checkpoint"),
            Path("/input"),
            Path("/output"),
            ["4", "5"],
        )
        self.assertIn("--nproc-per-node=2", cmd)
        self.assertIn("--model-mode=FORWARD_DYNAMICS", cmd)
        self.assertEqual(cmd[cmd.index("--num-steps") + 1], "4")
        self.assertEqual(cmd[cmd.index("--dp-shard-size") + 1], "2")
        self.assertEqual(cmd[cmd.index("--kv-cache-inference-size") + 1], "64")

    def test_transfer_retains_guardrails_and_training_aligned_control(self):
        cmd = runner.build_command(
            "transfer",
            Path("/framework"),
            Path("/checkpoint"),
            Path("/input"),
            Path("/output"),
            ["0"],
            Path("/vae"),
            Path("/tokenizer"),
            long=True,
        )
        self.assertNotIn("--no-guardrails", cmd)
        self.assertIn("model.config.teacher_forcing_frames_per_chunk=1", cmd)
        self.assertEqual(cmd[cmd.index("--kv-cache-inference-size") + 1], "30")


class PublicAuditTests(unittest.TestCase):
    def test_large_runtime_is_ignored_but_public_large_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "storage").mkdir()
            for path in [root / "storage/weights.bin", root / "preview.mp4"]:
                with path.open("wb") as stream:
                    stream.truncate(11 * 1024 * 1024)
            result = audit.audit(root)
            self.assertEqual(result["files"], 1)
            self.assertTrue(any("exceeds" in s for s in result["errors"]))


if __name__ == "__main__":
    unittest.main()
