"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Optional
import requests
import os
import tqdm
import io
from pathlib import Path
import torch

BASE_URL = "https://github.com/facebookresearch/nougat/releases/download"
MODEL_TAG = "0.1.0-small"
MODEL_TAG_BASE = "0.1.0-base"  # 与本仓库推理/微调兼容，推荐统一使用

# 下载超时（秒）：(连接超时, 读取超时)。大文件可设大一些或通过环境变量覆盖
DOWNLOAD_TIMEOUT = (30, 600)  # 连接 30s，读取 600s（10 分钟）


def _get_download_base_url() -> str:
    """若设置 NOUGAT_DOWNLOAD_MIRROR，则用镜像前缀（如国内 ghproxy），否则用 GitHub。"""
    mirror = os.environ.get("NOUGAT_DOWNLOAD_MIRROR", "").rstrip("/")
    if mirror:
        return f"{mirror}/{BASE_URL}" if mirror else BASE_URL
    return BASE_URL


# source: https://stackoverflow.com/a/71459251
def download_as_bytes_with_progress(url: str, name: str = None, timeout: tuple = None) -> bytes:
    """
    Download a file from a URL and return the contents as bytes, with progress bar.

    Args:
        url: The URL of the file to download.
        name: The name of the file to save to. If None, the filename will be the same as the URL.
        timeout: (connect_timeout, read_timeout) in seconds. Uses DOWNLOAD_TIMEOUT if None.

    Returns:
        bytes: The contents of the file.
    """
    timeout = timeout or DOWNLOAD_TIMEOUT
    # requests 会自动读取 HTTP_PROXY / HTTPS_PROXY 环境变量
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
    total = int(resp.headers.get("content-length", 0))
    bio = io.BytesIO()
    if name is None:
        name = url
    with tqdm.tqdm(
        desc=name,
        total=total,
        unit="b",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            bar.update(len(chunk))
            bio.write(chunk)
    return bio.getvalue()


def download_checkpoint(checkpoint: Path, model_tag: str = MODEL_TAG):
    """
    Download the Nougat model checkpoint.

    This function downloads the Nougat model checkpoint from GitHub.

    Args:
        checkpoint (Path): The path to the checkpoint.
        model_tag (str): The model tag to download. Default is "0.1.0-small".
    """
    base_url = _get_download_base_url()
    print("downloading nougat checkpoint version", model_tag, "to path", checkpoint)
    if os.environ.get("NOUGAT_DOWNLOAD_MIRROR"):
        print("using mirror:", os.environ.get("NOUGAT_DOWNLOAD_MIRROR"))
    files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for file in files:
        download_url = f"{base_url}/{model_tag}/{file}"
        binary_file = download_as_bytes_with_progress(download_url, file)
        if len(binary_file) > 15:  # sanity check
            (checkpoint / file).write_bytes(binary_file)


def torch_hub(model_tag: Optional[str] = MODEL_TAG) -> Path:
    old_path = Path(torch.hub.get_dir() + "/nougat")
    if model_tag is None:
        model_tag = MODEL_TAG
    hub_path = old_path.with_name(f"nougat-{model_tag}")
    if old_path.exists():
        # move to new format
        old_path.rename(old_path.with_name("nougat-0.1.0-small"))
    return hub_path


def get_checkpoint(
    checkpoint_path: Optional[os.PathLike] = None,
    model_tag: str = None,
    download: bool = True,
) -> Path:
    """
    Get the path to the Nougat model checkpoint.

    This function retrieves the path to the Nougat model checkpoint. If the checkpoint does not
    exist or is empty, it can optionally download the checkpoint.

    Args:
        checkpoint_path (Optional[os.PathLike]): The path to the checkpoint. If not provided,
            it will check the "NOUGAT_CHECKPOINT" environment variable or use the default location.
            Default is None.
        model_tag (str): The model tag to download (e.g. "0.1.0-base"). If None, uses env NOUGAT_MODEL_TAG or "0.1.0-small".
        download (bool): Whether to download the checkpoint if it doesn't exist or is empty.
            Default is True.

    Returns:
        Path: The path to the Nougat model checkpoint.
    """
    tag = model_tag or os.environ.get("NOUGAT_MODEL_TAG", MODEL_TAG)
    checkpoint = Path(
        checkpoint_path or os.environ.get("NOUGAT_CHECKPOINT", torch_hub(tag))
    )
    if checkpoint.exists() and checkpoint.is_file():
        checkpoint = checkpoint.parent
    if download and (not checkpoint.exists() or len(os.listdir(checkpoint)) < 5):
        checkpoint.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint, model_tag=tag)
    return checkpoint


def download_base_checkpoint(save_dir: os.PathLike) -> Path:
    """
    将与本仓库兼容的 nougat-base（GitHub Release 0.1.0-base）下载到指定目录。
    推理与微调请统一使用此来源，勿用 HuggingFace 的 facebook/nougat-base 目录。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    download_checkpoint(save_dir, model_tag=MODEL_TAG_BASE)
    return save_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "base":
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ.get("NOUGAT_CHECKPOINT", "nougat-base"))
        download_base_checkpoint(out)
        print("Downloaded 0.1.0-base to", out.resolve())
    else:
        get_checkpoint()
