"""Small native NATTEN kernel probe on ONE allocated idle GPU; not full-model validation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--framework", type=Path, default=ROOT / "repos/cosmos-framework")
    args = p.parse_args()
    if not args.gpu.isdigit():
        p.error("Select one physical GPU index allocated to you")
    if args.output.exists():
        p.error("Use a new report path")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_gpus.py"), "--gpus", args.gpu],
        check=True,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["COSMOS_TRAINING"] = "0"
    sys.path.insert(0, str(args.framework.resolve()))
    import torch
    import natten
    from cosmos_framework.model.attention import attention, multi_dimensional_attention
    from cosmos_framework.model.attention.masks import CausalType

    report = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "natten": natten.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "scope": "Small representative native kernels only. Passing does not validate the full three_way model, KV cache, memory capacity or output.",
        "cases": [],
    }
    torch.manual_seed(2026)
    with torch.inference_mode():
        q = torch.randn((1, 128, 8, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        offsets = torch.tensor([0, 128], device="cuda", dtype=torch.int32)

        def dense():
            return attention(q, k, v, backend="natten")

        def varlen():
            return attention(
                q,
                k,
                v,
                backend="natten",
                is_causal=True,
                causal_type=CausalType.DontCare,
                cumulative_seqlen_Q=offsets,
                cumulative_seqlen_KV=offsets,
                max_seqlen_Q=128,
                max_seqlen_KV=128,
            )

        def sparse():
            return multi_dimensional_attention(
                q.reshape(1, 2, 8, 8, 8, 128),
                k.reshape(1, 2, 8, 8, 2, 128),
                v.reshape(1, 2, 8, 8, 2, 128),
                window_size=(2, 4, 4),
                is_causal=(True, False, False),
                backend="natten",
            )

        for name, fn in [
            ("dense_gqa", dense),
            ("causal_varlen_gqa", varlen),
            ("causal_local_3d_gqa", sparse),
        ]:
            try:
                output = fn()
                torch.cuda.synchronize()
                finite = bool(torch.isfinite(output).all())
                report["cases"].append(
                    {
                        "case": name,
                        "finite": finite,
                        "shape": list(output.shape),
                        "passed": finite,
                    }
                )
            except Exception as exc:
                report["cases"].append(
                    {
                        "case": name,
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                traceback.print_exc()
    report["passed"] = all(row["passed"] for row in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
