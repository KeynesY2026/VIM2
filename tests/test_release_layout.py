import json
import re
from pathlib import Path

from vim2 import models as model_catalog
from vim2.models import MODEL_SPECS, ModelId

ROOT = Path(__file__).resolve().parents[1]
CPU_MODEL_DIRECTORY = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
CPU_MODEL_UPSTREAM = (
    "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
)
CPU_MODEL_REVISION = "68818b2313fe77bd06f6a7c5068ff3ef59d02b8a"
CPU_MODEL_HASHES = {
    "conv_frontend.onnx": (
        "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e"
    ),
    "encoder.int8.onnx": (
        "60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9"
    ),
    "decoder.int8.onnx": (
        "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771"
    ),
    "tokenizer/merges.txt": (
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
    ),
    "tokenizer/tokenizer_config.json": (
        "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"
    ),
    "tokenizer/vocab.json": (
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    ),
}


def test_launcher_uses_global_python_without_bundled_environment() -> None:
    launcher = (ROOT / "start.bat").read_text(encoding="utf-8")

    assert 'cd /d "%~dp0"' in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "runtime\\site-packages" not in launcher
    assert ".venv" not in launcher
    assert "pip install" not in launcher.lower()
    assert "python -m vim2 %*" in launcher


def test_windowed_launcher_uses_hidden_windows_powershell() -> None:
    launcher = (ROOT / "Start.cmd").read_text(encoding="utf-8")
    script = (ROOT / "start.ps1").read_text(encoding="utf-8")

    assert "powershell.exe" in launcher.lower()
    assert "-WindowStyle Hidden" in launcher
    assert "pythonw.exe" in script
    assert "-m\", \"vim2\", \"--windowed\"" in script


def test_release_preparation_uses_fixed_local_model_snapshots() -> None:
    script = (ROOT / "tools" / "prepare-release.ps1").read_text(
        encoding="utf-8"
    )

    assert "models--Qwen--Qwen3-ASR-0.6B" in script
    assert "5eb144179a02acc5e5ba31e748d22b0cf3e303b0" in script
    assert "models--Qwen--Qwen3-ASR-1.7B" in script
    assert "7278e1e70fe206f11671096ffdd38061171dd6e5" in script
    assert "Test-SherpaModelComplete" in script
    assert "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25" in script
    assert "--target" not in script
    assert "runtime\\site-packages" not in script
    assert "pip install" not in script.lower()


def test_release_manifest_and_dependency_lock_agree() -> None:
    manifest = json.loads(
        (ROOT / "release-manifest.json").read_text(encoding="utf-8")
    )
    lock = (ROOT / "requirements.lock").read_text(
        encoding="utf-8"
    )

    assert manifest["runtime"]["qwen-asr"] == "0.0.6"
    assert "qwen-asr==0.0.6" in lock
    assert manifest["runtime"]["transformers"] == "4.57.6"
    assert "transformers==4.57.6" in lock
    assert manifest["runtime"]["bitsandbytes"] == "0.48.2"
    assert "bitsandbytes==0.48.2" in lock
    assert manifest["runtime"]["sherpa-onnx"] == "1.13.8"
    assert "sherpa-onnx==1.13.8" in lock

    cpu_spec = MODEL_SPECS[ModelId.CPU]
    cpu_release = manifest["models"]["qwen3-asr-0.6b-int8-cpu"]
    assert cpu_release == {
        "directory": f".models/{CPU_MODEL_DIRECTORY}",
        "model_id": CPU_MODEL_UPSTREAM,
        "precision": "ONNX INT8 (sherpa-onnx CPU provider)",
        "revision": CPU_MODEL_REVISION,
    }
    assert cpu_spec.directory_name == CPU_MODEL_DIRECTORY
    assert cpu_spec.upstream_id == cpu_release["model_id"] == CPU_MODEL_UPSTREAM
    assert cpu_spec.revision == cpu_release["revision"] == CPU_MODEL_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", cpu_spec.revision)
    assert "qwen3-asr-0.6b-int8" not in manifest["models"]


def test_model_checksum_manifest_is_utf8_json() -> None:
    checksums = json.loads(
        (ROOT / "release-files.sha256.json").read_text(encoding="utf-8")
    )

    assert checksums[".models\\Qwen3-ASR-0.6B\\model.safetensors"] == (
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
    )
    assert model_catalog._REQUIRED_SHERPA_FILES == tuple(CPU_MODEL_HASHES)
    cpu_prefix = f".models\\{CPU_MODEL_DIRECTORY}\\"
    approved_hashes = {
        name: checksums[cpu_prefix + name.replace("/", "\\")]
        for name in CPU_MODEL_HASHES
    }
    assert approved_hashes == CPU_MODEL_HASHES
