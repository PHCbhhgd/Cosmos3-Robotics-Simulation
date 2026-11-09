"""Download pinned public robot examples/weights without sending credentials.

curl identity encoding avoids the installed httpx2/Brotli incompatibility.
Large artifacts and download receipts stay under ignored storage/.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "15",
            "--max-time",
            "1800" if path.suffix == ".safetensors" else "90",
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "--header",
            "Accept-Encoding: identity",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    with partial.open("rb") as stream:
        prefix = stream.read(64)
    if not prefix or prefix.startswith(b"version https://git-lfs.github.com"):
        raise ValueError(f"Empty download or Git LFS pointer: {path.name}")
    partial.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["bimanual", "transfer"])
    p.add_argument("--weights", action="store_true")
    p.add_argument(
        "--long", action="store_true", help="Also fetch Transfer example1 (20 seconds)"
    )
    p.add_argument("--endpoint", default="https://huggingface.co")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    endpoint = args.endpoint.rstrip("/")
    if urlsplit(endpoint).scheme != "https" or urlsplit(endpoint).netloc not in {
        "huggingface.co",
        "hf-mirror.com",
    }:
        p.error("Use https://huggingface.co or https://hf-mirror.com")
    model = json.loads((ROOT / "configs/versions.json").read_text())["models"][
        args.model
    ]
    dest = ROOT / "storage/models" / model["repo_id"].split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    api = dest / ".public-api.json"
    if not api.exists():
        download(
            f"{endpoint}/api/models/{model['repo_id']}/revision/{model['revision']}",
            api,
        )
    meta = json.loads(api.read_text())
    if meta["sha"] != model["revision"]:
        raise ValueError("API did not return pinned revision")

    def selected(name):
        if name == "assets/training/agibot/gt/agibot_001_gt.mp4":
            return True
        if name.startswith(("assets/sim_bimanual/", "examples/inference/")):
            return True
        if name.startswith("assets/sample_data/"):
            return args.long or "/example2/" in name
        if name.startswith(
            (
                "cosmos3_nano_sim_bimanual_action_sources/",
                "cosmos_dreams_action_sources/",
            )
        ):
            return True
        if "/" not in name and name.endswith((".json", ".md")):
            return True
        if name in {
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
        }:
            return True
        return (
            args.weights
            and name.startswith(("transformer/", "vae/"))
            and name.endswith(".safetensors")
        )

    def fetch(name):
        path = dest / name
        if not path.exists():
            download(
                f"{endpoint}/{model['repo_id']}/resolve/{model['revision']}/{quote(name, safe='/')}?download=true",
                path,
            )
        result = {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        print(json.dumps(result), flush=True)
        return result

    files = sorted(f["rfilename"] for f in meta["siblings"] if selected(f["rfilename"]))
    errors, results = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {name: pool.submit(fetch, name) for name in files}
        for name, future in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"path": name, "error": str(exc)})
    receipt = {
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "endpoint": endpoint,
        "credential_sent": False,
        "weights_requested": args.weights,
        "files": results,
        "errors": errors,
    }
    (dest / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(int(bool(errors)))


if __name__ == "__main__":
    main()
