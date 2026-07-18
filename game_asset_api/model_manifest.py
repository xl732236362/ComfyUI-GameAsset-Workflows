"""Verified model specifications for the pixel game asset workflow."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256 as sha256_constructor
from hmac import compare_digest
from pathlib import Path
import os
import re
import subprocess


_HASH_CHUNK_SIZE = 1024 * 1024
_RETRY_ALL_ERRORS_MINIMUM_VERSION = (7, 71, 0)
_CURL_VERSION_PATTERN = re.compile(r"^curl (\d+)\.(\d+)\.(\d+)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    filename: str
    relative_dir: str
    url: str
    size: int
    sha256: str
    fallback_urls: tuple[str, ...] = ()

    def destination(self, root: Path) -> Path:
        return root / "models" / self.relative_dir / self.filename


MODEL_SPECS = (
    ModelSpec(
        filename="sd_xl_base_1.0.safetensors",
        relative_dir="checkpoints",
        url="https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        size=6_938_078_334,
        sha256="31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
    ),
    ModelSpec(
        filename="pixel-art-xl.safetensors",
        relative_dir="loras",
        url="https://hf-mirror.com/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors",
        size=170_543_052,
        sha256="4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7",
    ),
    ModelSpec(
        filename="BiRefNet-general-epoch_244.safetensors",
        relative_dir="background_removal",
        url="https://hf-mirror.com/ZhengPeng7/BiRefNet/resolve/main/model.safetensors",
        size=444_473_596,
        sha256="9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
    ),
    ModelSpec(
        filename="OpenPoseXL2.safetensors",
        relative_dir="controlnet",
        url="https://hf-mirror.com/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors",
        size=5_004_167_829,
        sha256="5a4b928cb1e93748217900cb66d4135bf70d932d2924232f925910fad9e43a92",
    ),
    ModelSpec(
        filename="ip-adapter-plus_sdxl_vit-h.safetensors",
        relative_dir="ipadapter",
        url="https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        size=847_517_512,
        sha256="3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581",
    ),
    ModelSpec(
        filename="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        relative_dir="clip_vision",
        url="https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
        size=2_528_373_448,
        sha256="6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
    ),
    ModelSpec(
        filename="mm_sdxl_v10_beta.safetensors",
        relative_dir="animatediff_models",
        url=(
            "https://hf-mirror.com/guoyww/animatediff-motion-adapter-sdxl-beta/resolve/"
            "26c864717b4d4b002bb48ae6c9d6bb431548c6cb/diffusion_pytorch_model.fp16.safetensors"
        ),
        size=474_328_896,
        sha256="24c3c5f48006ce2ce7b06188622865c620b2d33db23b1af671cc1f21716b5826",
        fallback_urls=(
            "https://huggingface.co/guoyww/animatediff-motion-adapter-sdxl-beta/resolve/"
            "26c864717b4d4b002bb48ae6c9d6bb431548c6cb/diffusion_pytorch_model.fp16.safetensors",
        ),
    ),
)


def verify_file(path: Path, size: int, sha256: str) -> bool:
    """Return whether *path* is a complete file matching the expected digest."""
    if not path.is_file() or path.stat().st_size != size:
        return False

    digest = sha256_constructor()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return compare_digest(digest.hexdigest(), sha256)


def _curl_supports_retry_all_errors() -> bool:
    try:
        version = subprocess.run(
            ["curl.exe", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    match = _CURL_VERSION_PATTERN.search(version.stdout or "")
    if match is None:
        return False
    return tuple(map(int, match.groups())) >= _RETRY_ALL_ERRORS_MINIMUM_VERSION


def install(spec: ModelSpec, root: Path) -> Path:
    """Download *spec* to *root* and promote it only after verification."""
    destination = spec.destination(root)
    if verify_file(destination, spec.size, spec.sha256):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if verify_file(partial, spec.size, spec.sha256):
        os.replace(partial, destination)
        return destination
    if partial.exists() and partial.stat().st_size >= spec.size:
        partial.unlink()

    last_error = None
    for url in (spec.url, *spec.fallback_urls):
        command = [
            "curl.exe",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "10",
        ]
        if _curl_supports_retry_all_errors():
            command.append("--retry-all-errors")
        command.extend(["--output", str(partial), url])
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            last_error = error
            continue
        if not verify_file(partial, spec.size, spec.sha256):
            raise RuntimeError(f"Downloaded model failed verification: {spec.filename}")

        os.replace(partial, destination)
        return destination

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Downloaded model failed verification: {spec.filename}")
