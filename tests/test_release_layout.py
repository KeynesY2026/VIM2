import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_windowed_launcher_uses_hidden_windows_powershell() -> None:
    launcher = (ROOT / "Start.cmd").read_text(encoding="utf-8")
    script = (ROOT / "start.ps1").read_text(encoding="utf-8")

    assert "powershell.exe" in launcher.lower()
    assert "-WindowStyle Hidden" in launcher
    assert "pythonw.exe" in script
    assert "-m\", \"vim2\", \"--windowed\"" in script


@pytest.mark.skipif(
    os.name != "nt" or shutil.which("powershell.exe") is None,
    reason="Requires Windows PowerShell and pythonw.exe",
)
def test_launcher_exits_without_waiting_for_the_windowed_app(tmp_path: Path) -> None:
    root = tmp_path / "portable app with spaces"
    module = root / "app" / "vim2"
    module.mkdir(parents=True)
    shutil.copyfile(ROOT / "start.ps1", root / "start.ps1")
    (module / "__main__.py").write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[2]\n"
        "(root / 'started.json').write_text(json.dumps({\n"
        "    'cwd': os.getcwd(), 'args': sys.argv[1:],\n"
        "    'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "    'hf_home': os.environ.get('HF_HOME'),\n"
        "    'offline': os.environ.get('HF_HUB_OFFLINE'),\n"
        "    'transformers_offline': os.environ.get('TRANSFORMERS_OFFLINE'),\n"
        "}), encoding='utf-8')\n"
        "deadline = time.monotonic() + 60\n"
        "while not (root / 'release').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "(root / 'finished').touch()\n",
        encoding="utf-8",
    )
    python = sys.executable.replace("'", "''")
    script = str(root / "start.ps1").replace("'", "''")
    harness = root / "launch.ps1"
    harness.write_text(
        "function Get-Command {\n"
        "    [CmdletBinding()]\n"
        "    param([string]$Name)\n"
        f"    [pscustomobject]@{{ Source = '{python}' }}\n"
        "}\n"
        f"& '{script}'\n",
        encoding="utf-8",
    )

    def wait_for_file(path: Path) -> None:
        deadline = time.monotonic() + 10
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert path.exists(), f"Launcher child did not create {path.name}"

    with (root / "launcher.log").open("w", encoding="utf-8") as log:
        launcher = subprocess.Popen(
            [
                "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File", str(harness),
            ],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            assert launcher.wait(timeout=10) == 0
            wait_for_file(root / "started.json")
            assert not (root / "finished").exists()
            started = json.loads((root / "started.json").read_text(encoding="utf-8"))
            assert Path(started["cwd"]) == root
            assert started["args"] == ["--windowed"]
            assert Path(started["pythonpath"]) == root / "app"
            assert Path(started["hf_home"]) == root / "temp" / "huggingface"
            assert started["offline"] == "1"
            assert started["transformers_offline"] == "1"
        finally:
            (root / "release").touch()
            launcher.wait(timeout=10)
            if (root / "started.json").exists():
                wait_for_file(root / "finished")


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
    assert "requirements-gpu-cu128.lock" in script


def test_installation_documents_cpu_and_gpu_dependency_profiles() -> None:
    instructions = (ROOT / "INSTALLATION.md").read_text(encoding="utf-8")

    assert "requirements-cpu.lock" in instructions
    assert "requirements-gpu-cu128.lock" in instructions
    assert "requirements.lock" not in instructions
    assert not (ROOT / "requirements.lock").exists()
    assert not (ROOT / "requirements-cpu.txt").exists()


def test_cpu_and_gpu_dependency_locks_are_separate() -> None:
    cpu_lock = (ROOT / "requirements-cpu.lock").read_text(encoding="utf-8")
    gpu_lock = (ROOT / "requirements-gpu-cu128.lock").read_text(
        encoding="utf-8"
    )

    assert "PySide6==6.10.0" in cpu_lock
    assert "sherpa-onnx==1.13.8" in cpu_lock
    assert "torch==" not in cpu_lock
    assert "bitsandbytes==" not in cpu_lock
    assert "qwen-asr==" not in cpu_lock
    assert "torch==2.11.0+cu128" in gpu_lock
    assert "bitsandbytes==0.48.2" in gpu_lock
    assert "qwen-asr==0.0.6" in gpu_lock
    assert "webrtcvad-wheels==2.0.14.post1" in gpu_lock


def test_release_manifest_and_dependency_lock_agree() -> None:
    manifest = json.loads(
        (ROOT / "release-manifest.json").read_text(encoding="utf-8")
    )
    lock = (ROOT / "requirements-gpu-cu128.lock").read_text(
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
    assert manifest["models"]["qwen3-asr-0.6b-int8-cpu"] == {
        "directory": (
            ".models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
        ),
        "model_id": "zengshuishui/Qwen3-ASR-onnx",
        "precision": "ONNX INT8 (sherpa-onnx CPU provider)",
        "revision": "2026-03-25",
    }
    assert "qwen3-asr-0.6b-int8" not in manifest["models"]


def test_model_checksum_manifest_is_utf8_json() -> None:
    checksums = json.loads(
        (ROOT / "release-files.sha256.json").read_text(encoding="utf-8")
    )

    assert checksums[".models\\Qwen3-ASR-0.6B\\model.safetensors"] == (
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
    )
    assert checksums[
        ".models\\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
        "\\decoder.int8.onnx"
    ] == "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771"
