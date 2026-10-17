"""Generate src/parley/cuda_pack.json: the CUDA 12 libraries a frozen Parley downloads on first run (NVIDIA GPUs only).

    uv run python scripts/cuda_pack.py            # rewrite src/parley/cuda_pack.json from uv.lock
    uv run python scripts/cuda_pack.py --check    # exit 1 if the file differs from what uv.lock implies

URL, sha256 and size of each wheel come straight from `uv.lock` (win_amd64 wheel of each `nvidia-*-cu12` package), so the
pack is exactly the CUDA stack the source install runs with. scripts/build.py regenerates the file before bundling it;
tests/test_cuda_pack.py fails if the committed copy drifts from the lock.

Which DLLs: measured on the source install (RTX 5090, CUDA toolkit removed from PATH, Whisper + Kokoro on the GPU, loaded
modules listed with EnumProcessModules). Loaded: cublas64_12, cublasLt64_12, cudart64_12, cufft64_11, cudnn64_9 and the
cudnn engine/graph/ops/adv/heuristic libraries. Also included because they are LoadLibrary'd by name on demand (cuDNN
sub-libraries, nvrtc for cuDNN runtime-compiled engines, nvJitLink for cuFFT): cudnn_cnn/ext, nvrtc, nvrtc-builtins,
nvJitLink. Left out because nothing loads them: nvblas64_12 (BLAS interposer), cufftw64_11 (FFTW shim),
nvrtc64_120_0.alt.dll (alternate nvrtc). A new `nvidia-*-cu12` package in the lock stops the generator until it is added
to DLLS here on purpose.
"""
import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "uv.lock"
OUT = ROOT / "src" / "parley" / "cuda_pack.json"
ALLOWED_HOSTS = ("files.pythonhosted.org",)
WHEEL_TAG = "win_amd64"

# package -> (directory inside the wheel's `nvidia/` tree, DLL file names to extract)
DLLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "nvidia-cublas-cu12": ("cublas", ("cublas64_12.dll", "cublasLt64_12.dll")),
    "nvidia-cuda-nvrtc-cu12": ("cuda_nvrtc", ("nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll")),
    "nvidia-cuda-runtime-cu12": ("cuda_runtime", ("cudart64_12.dll",)),
    "nvidia-cudnn-cu12": ("cudnn", (
        "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll", "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll", "cudnn_engines_tensor_ir64_9.dll", "cudnn_ext64_9.dll",
        "cudnn_graph64_9.dll", "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll")),
    "nvidia-cufft-cu12": ("cufft", ("cufft64_11.dll",)),
    "nvidia-nvjitlink-cu12": ("nvjitlink", ("nvJitLink_120_0.dll",)),
}
_NVIDIA_CU12 = re.compile(r"nvidia-.*-cu12")


class PackError(Exception):
    pass


def build_manifest(lock_text: str) -> dict:
    """The manifest dict for a uv.lock text. Raises PackError on anything unexpected."""
    packages = {p["name"]: p for p in tomllib.loads(lock_text).get("package", [])}
    present = {n for n in packages if _NVIDIA_CU12.fullmatch(n)}
    if present != set(DLLS):
        raise PackError(f"uv.lock nvidia-*-cu12 packages {sorted(present)} differ from the DLL table {sorted(DLLS)}: "
                        "decide which DLLs the new/removed package contributes and edit DLLS in scripts/cuda_pack.py")
    wheels = []
    for name in sorted(DLLS):
        pkg, (subdir, dlls) = packages[name], DLLS[name]
        hits = [w for w in pkg.get("wheels", []) if urlsplit(w["url"]).path.endswith(f"-{WHEEL_TAG}.whl")]
        if len(hits) != 1:
            raise PackError(f"{name}: expected one {WHEEL_TAG} wheel in uv.lock, found {len(hits)}")
        w = hits[0]
        if urlsplit(w["url"]).scheme != "https" or urlsplit(w["url"]).hostname not in ALLOWED_HOSTS:
            raise PackError(f"{name}: wheel URL is not https on {ALLOWED_HOSTS}")
        digest = w.get("hash", "")
        if not digest.startswith("sha256:") or not re.fullmatch(r"[0-9a-f]{64}", digest[7:]):
            raise PackError(f"{name}: no sha256 in uv.lock")
        wheels.append({
            "package": name, "version": pkg["version"], "url": w["url"], "sha256": digest[7:], "size": int(w["size"]),
            "dlls": [f"nvidia/{subdir}/bin/{d}" for d in dlls],
        })
    return {"schema": 1, "platform": WHEEL_TAG, "allowed_hosts": list(ALLOWED_HOSTS), "wheels": wheels}


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def generate(lock: Path = LOCK, out: Path = OUT) -> bool:
    """Write `out` from `lock`; True if the file changed."""
    text = render(build_manifest(lock.read_text("utf-8")))
    old = out.read_text("utf-8") if out.exists() else None
    if old != text:
        out.write_text(text, "utf-8", newline="\n")
    return old != text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only compare; exit 1 if the committed file is out of date")
    args = ap.parse_args()
    try:
        text = render(build_manifest(LOCK.read_text("utf-8")))
    except PackError as e:
        print(f"cuda_pack: {e}", file=sys.stderr)
        return 1
    current = OUT.read_text("utf-8") if OUT.exists() else None
    if args.check:
        if current != text:
            print(f"cuda_pack: {OUT} is out of date; run scripts/cuda_pack.py", file=sys.stderr)
            return 1
        return 0
    if current != text:
        OUT.write_text(text, "utf-8", newline="\n")
    total = sum(w["size"] for w in json.loads(text)["wheels"])
    print(f"{OUT.name}: {len(json.loads(text)['wheels'])} wheels, {total / 1e6:.0f} MB download, "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
