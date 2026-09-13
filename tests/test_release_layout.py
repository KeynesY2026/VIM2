import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def test_model_checksum_manifest_is_utf8_json() -> None:
    checksums = json.loads(
        (ROOT / "release-files.sha256.json").read_text(encoding="utf-8")
    )

    assert checksums[".models\\Qwen3-ASR-0.6B\\model.safetensors"] == (
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
    )
