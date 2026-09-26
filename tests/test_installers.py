"""Installer boundary regression tests; no GUI launch or real user profile I/O."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vim2.paths import AppPaths

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_paths_and_first_run_are_separate(tmp_path, monkeypatch):
    bundle = tmp_path / "VIM2 bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "settings.json").write_text('{"selected_model":"qwen3-asr-0.6b-int8-cpu"}')
    (bundle / "config" / "hotkey.conf").write_text("RightAlt")
    (bundle / "config" / "hotwords.template.txt").write_text("original")
    (bundle / "config" / "hotwords.txt").write_text("PRIVATE - must not be copied")
    user = tmp_path / "user"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    paths = AppPaths.for_launch(data_root=user)
    assert paths.root == bundle
    assert paths.models_dir == bundle / ".models"
    assert paths.config_dir == user / "config"
    assert paths.runtime_dir == user / "runtime"
    assert paths.temp_dir == user / "temp"
    paths.initialize_user_data()
    (paths.config_dir / "settings.json").write_text("custom")
    (paths.config_dir / "hotwords.txt").write_text("custom hotwords")
    paths.initialize_user_data()
    assert (paths.config_dir / "settings.json").read_text() == "custom"
    assert paths.hotwords_file.read_text() == "custom hotwords"
    assert (paths.config_dir / "hotkey.conf").read_text() == "RightAlt"
    assert paths.runtime_dir.is_dir() and paths.temp_dir.is_dir()
    assert not (bundle / "runtime").exists()


def test_frozen_data_root_platform_selection(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert AppPaths.for_launch(platform="win32").config_dir == tmp_path / "local" / "VIM2" / "config"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert AppPaths.for_launch(platform="darwin").runtime_dir == tmp_path / "home" / "Library" / "Application Support" / "VIM2" / "runtime"


def test_frozen_restart_uses_executable_not_python_module(tmp_path, monkeypatch):
    from vim2 import qt_runtime
    calls = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/Applications/VIM2.app/Contents/MacOS/VIM2")
    monkeypatch.setattr(qt_runtime.os, "execv", lambda exe, argv: calls.append((exe, argv)))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    qt_runtime._restart_process(AppPaths.for_launch())
    assert calls == [(sys.executable, [sys.executable, "--windowed"])]


def test_frozen_main_check_uses_bundle_and_never_portable_tree(tmp_path, monkeypatch, capsys):
    from vim2 import main as main_module
    from vim2.models import MODEL_SPECS, ModelId, _REQUIRED_SHERPA_FILES
    from vim2.platform_services import select_platform_profile
    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "settings.json").write_text('{"selected_model":"qwen3-asr-0.6b-int8-cpu"}')
    (bundle / "config" / "hotkey.conf").write_text('RightAlt')
    (bundle / "config" / "hotwords.template.txt").write_text('# default\n')
    model = bundle / '.models' / MODEL_SPECS[ModelId.CPU].directory_name
    for name in _REQUIRED_SHERPA_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('stub')
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(bundle), raising=False)
    monkeypatch.setattr(main_module, 'select_platform_profile', lambda: select_platform_profile('win32', 'AMD64'))
    data = tmp_path / 'data'
    assert main_module.run(['--check', '--skip-runtime-check', '--data-root', str(data)]) == 0
    assert (data / 'runtime' / 'vim2.log').is_file()
    assert (data / 'config' / 'settings.json').is_file()
    assert (data / 'config' / 'hotwords.txt').read_text() == '# default\n'
    assert main_module.run(['--root', str(tmp_path), '--check', '--data-root', str(data)]) == 1
    assert 'Frozen builds do not accept --root' in capsys.readouterr().err


def test_cpu_check_loads_native_runtime_and_reports_import_errors(tmp_path, monkeypatch, capsys):
    from vim2 import main as main_module
    from vim2.models import ModelId
    from types import SimpleNamespace
    import vim2.application as application_module
    seen = []
    class Checker:
        def __init__(self, paths, *, platform_profile):
            assert paths.root == tmp_path.resolve()
        def check(self, selected_model, **kwargs):
            assert selected_model == ModelId.CPU
            return SimpleNamespace(ok=True, errors=())
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'settings.json').write_text('{"selected_model":"qwen3-asr-0.6b-int8-cpu"}')
    monkeypatch.setattr(main_module, 'PreflightChecker', Checker)
    monkeypatch.setattr(main_module, 'create_platform_services', lambda *a, **kw: SimpleNamespace(preflight_errors=lambda: ()))
    monkeypatch.setattr(main_module, 'select_platform_profile', lambda: SimpleNamespace(name='windows'))
    def unavailable(name):
        seen.append(name)
        if name == 'sounddevice':
            raise ImportError('native device library missing')
        return object()
    monkeypatch.setattr(main_module.importlib, 'import_module', unavailable)
    assert main_module.run(['--root', str(tmp_path), '--check']) == 1
    assert seen == ['sherpa_onnx', 'sounddevice']
    assert 'native device library missing' in capsys.readouterr().err


def test_payload_verifier_rejects_private_and_developer_paths(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('payload_verifier', ROOT / 'tools' / 'verify-installer-layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / '.squad').mkdir()
    (tmp_path / '.squad' / 'secret.txt').write_text('never package')
    with pytest.raises(RuntimeError, match='Forbidden payload path'):
        module.verify(tmp_path)
    (tmp_path / '.squad' / 'secret.txt').unlink()
    (tmp_path / '.squad').rmdir()
    (tmp_path / 'internal.h').write_text('header')
    with pytest.raises(RuntimeError, match='Developer file'):
        module.verify(tmp_path)


def test_installer_inputs_are_explicit_and_cpu_only():
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    assert manifest["installers"]["model"] == "qwen3-asr-0.6b-int8-cpu"
    spec = (ROOT / "packaging" / "vim2.spec").read_text()
    assert "_REQUIRED_SHERPA_FILES" in spec and "collect_dynamic_libs('sherpa_onnx')" in spec
    assert "'sounddevice', 'soundfile'" in spec
    assert "_MEIPASS" not in spec
    assert "squad" not in spec.lower() and "runtime/" not in spec
    workflow = (ROOT / ".github/workflows/build-installers.yml").read_text()
    assert "windows-2022" in workflow and "workflow_dispatch:" in workflow
    assert "68818b2313fe77bd06f6a7c5068ff3ef59d02b8a" in workflow
    assert "secrets." not in workflow
    iss = (ROOT / "packaging" / "vim2.iss").read_text()
    assert "PrivilegesRequired=lowest" in iss and "VIM2.exe" in iss
    lock = (ROOT / "requirements-windows-cpu.lock").read_text().lower()
    for excluded in ("torch", "cuda", "pyobjc", "bitsandbytes", "qwen-asr"):
        assert excluded not in lock


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "payload_verifier",
        ROOT / "tools" / "verify-installer-layout.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_hashes(payload: dict[str, bytes]) -> dict[str, str]:
    return {
        "\\".join(Path(relative).parts): hashlib.sha256(content).hexdigest()
        for relative, content in payload.items()
        if relative.startswith(".models/")
    }


def _write_payload(root: Path, payload: dict[str, bytes]) -> None:
    for relative, content in payload.items():
        path = root.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _valid_payload() -> dict[str, bytes]:
    model = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    files = {
        f".models/{model}/conv_frontend.onnx": b"frontend",
        f".models/{model}/encoder.int8.onnx": b"encoder",
        f".models/{model}/decoder.int8.onnx": b"decoder",
        f".models/{model}/tokenizer/merges.txt": b"merges",
        f".models/{model}/tokenizer/tokenizer_config.json": b"{}",
        f".models/{model}/tokenizer/vocab.json": b"[]",
        "config/settings.json": b'{"selected_model":"qwen3-asr-0.6b-int8-cpu"}\n',
        "config/hotkey.conf": b"RightAlt\n",
        "config/hotwords.template.txt": b"# template\n",
    }
    return files


def _bundle_with_defaults(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "settings.json").write_text(
        '{"selected_model":"qwen3-asr-0.6b-int8-cpu"}'
    )
    (bundle / "config" / "hotkey.conf").write_text("RightAlt\n")
    (bundle / "config" / "hotwords.template.txt").write_text("# default\n")
    return bundle


def test_payload_allowlist_accepts_exact_files_and_ignores_dependency_trees(tmp_path):
    module = _verifier()
    payload = _valid_payload()
    _write_payload(tmp_path, payload)
    decoy = tmp_path / "numpy" / "config"
    decoy.mkdir(parents=True)
    (decoy / "settings.json").write_text("dependency config must not be owned")
    nested_models = tmp_path / "sherpa_onnx" / ".models"
    nested_models.mkdir(parents=True)
    (nested_models / "weights.onnx").write_bytes(b"not the product model")
    module.verify(tmp_path, hashes=_fixture_hashes(payload))


def test_payload_allowlist_rejects_extra_model_config_case_symlink_and_hash(tmp_path):
    module = _verifier()
    payload = _valid_payload()
    root = tmp_path / "payload"
    _write_payload(root, payload)
    hashes = _fixture_hashes(payload)
    extra_model = root / ".models" / "Qwen3-ASR-0.6B" / "model.safetensors"
    extra_model.parent.mkdir()
    extra_model.write_bytes(b"gpu")
    with pytest.raises(RuntimeError, match="Extra model or config"):
        module.verify(root, hashes=hashes)
    extra_model.unlink()
    extra_model.parent.rmdir()

    (root / "config" / "private.json").write_text("secret")
    with pytest.raises(RuntimeError, match="Extra model or config"):
        module.verify(root, hashes=hashes)
    (root / "config" / "private.json").unlink()

    (root / "config" / "hotwords.txt").write_text("PRIVATE")
    with pytest.raises(RuntimeError, match="Private hotwords"):
        module.verify(root, hashes=hashes)
    (root / "config" / "hotwords.txt").unlink()

    wrong_case = tmp_path / "case"
    wrong_case.mkdir()
    _write_payload(wrong_case, payload)
    settings = wrong_case / "config" / "settings.json"
    settings.rename(wrong_case / "config" / "Settings.json")
    with pytest.raises(RuntimeError, match="Case variant"):
        module.verify(wrong_case, hashes=hashes)

    escaped = tmp_path / "outside.txt"
    escaped.write_text("outside")
    linked = root / "config" / "escaped.conf"
    linked.symlink_to(escaped)
    with pytest.raises(RuntimeError, match="Symlink"):
        module.verify(root, hashes=hashes)
    linked.unlink()

    tampered = root / ".models" / module.MODEL / "decoder.int8.onnx"
    tampered.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        module.verify(root, hashes=hashes)


def test_resource_root_does_not_treat_framework_mirror_as_second_payload(tmp_path):
    module = _verifier()
    payload = _valid_payload()
    app = tmp_path / "VIM2.app"
    resources = app / "Contents" / "Resources"
    _write_payload(resources, payload)
    frameworks = app / "Contents" / "Frameworks"
    frameworks.mkdir(parents=True)
    (frameworks / ".models").symlink_to(resources / ".models")
    (frameworks / "config").symlink_to(resources / "config")
    dependency = frameworks / "numpy" / "config"
    dependency.mkdir(parents=True)
    (dependency / "settings.json").write_text("library")
    module.verify(app, hashes=_fixture_hashes(payload))
    outside = tmp_path / "escaped-models"
    outside.mkdir()
    (frameworks / "evil.models").symlink_to(outside)
    # A non-canonical project-tree symlink must not be accepted.
    (frameworks / ".models").unlink()
    (frameworks / ".models").symlink_to(outside)
    with pytest.raises(RuntimeError, match="escapes resource root"):
        module.verify(app, hashes=_fixture_hashes(payload))


def test_source_payload_rejects_symlink_hash_and_private_selection(tmp_path):
    module = _verifier()
    payload = _valid_payload()
    repo = tmp_path / "repo"
    _write_payload(repo, payload)
    template = repo / "config" / "hotwords.template.txt"
    packaged = repo / "packaging"
    packaged.mkdir()
    (packaged / "hotwords.template.txt").write_bytes(template.read_bytes())
    template.unlink()
    (repo / "config" / "hotwords.txt").write_text("PRIVATE - not packaged")
    (repo / ".models" / "Qwen3-ASR-0.6B").mkdir()
    (repo / ".models" / "Qwen3-ASR-0.6B" / "model.safetensors").write_bytes(b"gpu")
    (repo / ".models" / module.MODEL / "README.md").write_text("not packaged")
    selected = module.packaged_source_files(repo)
    assert set(selected) == set(module.PACKAGED_PATHS)
    assert "config/hotwords.txt" not in selected
    assert not any("Qwen3-ASR-0.6B" in path for path in selected)
    module.verify_source_payload(repo, hashes=_fixture_hashes(payload))

    model_file = repo / ".models" / module.MODEL / "encoder.int8.onnx"
    target = model_file.read_bytes()
    model_file.unlink()
    outside = tmp_path / "linked-encoder"
    outside.write_bytes(target)
    model_file.symlink_to(outside)
    with pytest.raises(RuntimeError, match="Symlink"):
        module.verify_source_payload(repo, hashes=_fixture_hashes(payload))
    model_file.unlink()
    model_file.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        module.verify_source_payload(repo, hashes=_fixture_hashes(payload))


def test_release_hash_keys_use_exact_allowlisted_model_paths():
    module = _verifier()
    hashes = json.loads((ROOT / "release-files.sha256.json").read_text(encoding="utf-8"))
    assert module.PACKAGED_CONFIG == (
        "config/settings.json",
        "config/hotkey.conf",
        "config/hotwords.template.txt",
    )
    assert len(module.MODEL_FILES) == 6
    for name in module.MODEL_FILES:
        key = module.hash_key(Path(".models") / module.MODEL / name)
        assert key in hashes
        assert len(hashes[key]) == 64


def test_first_run_partial_write_is_not_kept_and_retry_preserves_user_data(
    tmp_path, monkeypatch
):
    bundle = _bundle_with_defaults(tmp_path)
    user = tmp_path / "user"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    paths = AppPaths.for_launch(data_root=user)
    real_copy = shutil.copyfileobj
    calls = {"n": 0}

    def fail_second(source, destination, length=0):
        calls["n"] += 1
        if calls["n"] == 2:
            destination.write(b"PARTIAL")
            raise OSError("injected copy failure")
        return real_copy(source, destination, length)

    monkeypatch.setattr(shutil, "copyfileobj", fail_second)
    with pytest.raises(OSError, match="injected copy failure"):
        paths.initialize_user_data()
    config = user / "config"
    assert (config / "settings.json").read_text() == (
        '{"selected_model":"qwen3-asr-0.6b-int8-cpu"}'
    )
    assert not (config / "hotkey.conf").exists()
    assert not (config / "hotwords.txt").exists()
    assert not any(path.name == "hotkey.conf" for path in config.iterdir())

    monkeypatch.setattr(shutil, "copyfileobj", real_copy)
    paths.initialize_user_data()
    assert (config / "hotkey.conf").read_text() == "RightAlt\n"
    assert (config / "hotwords.txt").read_text() == "# default\n"
    (config / "settings.json").write_text("USER")
    (config / "hotwords.txt").write_text("USER WORDS")
    paths.initialize_user_data()
    assert (config / "settings.json").read_text() == "USER"
    assert (config / "hotwords.txt").read_text() == "USER WORDS"


def test_first_run_publishes_with_fsync_and_replace_and_replace_failure_retries(
    tmp_path, monkeypatch
):
    bundle = _bundle_with_defaults(tmp_path)
    user = tmp_path / "user"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    paths = AppPaths.for_launch(data_root=user)
    events: list[object] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def fail_replace(source, target):
        events.append(("replace", Path(source).name, Path(target).name))
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        paths.initialize_user_data()
    assert "fsync" in events
    assert events.index("fsync") < events.index(
        next(item for item in events if isinstance(item, tuple))
    )
    replace = next(item for item in events if isinstance(item, tuple))
    assert replace[2] == "settings.json"
    assert replace[1] != "settings.json"
    assert not (user / "config" / "settings.json").exists()
    assert not (user / "config" / "hotkey.conf").exists()

    def spy_replace(source, target):
        events.append(("replace-ok", Path(target).name))
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", spy_replace)
    paths.initialize_user_data()
    assert (user / "config" / "settings.json").read_text() == (
        '{"selected_model":"qwen3-asr-0.6b-int8-cpu"}'
    )
    (user / "config" / "settings.json").write_bytes(b"KEEP")

    def forbid_replace(source, target):
        raise AssertionError(f"must not replace existing user file {target}")

    monkeypatch.setattr(os, "replace", forbid_replace)
    paths.initialize_user_data()
    assert (user / "config" / "settings.json").read_bytes() == b"KEEP"


def test_import_smoke_uses_isolated_data_root_without_gui_or_tcc(
    tmp_path, monkeypatch, capsys
):
    from vim2 import main as main_module

    bundle = _bundle_with_defaults(tmp_path)
    data = tmp_path / "isolated"
    imported: list[str] = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(
        main_module.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("import smoke must not touch platform services or GUI")

    monkeypatch.setattr(main_module, "create_platform_services", forbidden)
    monkeypatch.setattr(main_module, "select_platform_profile", forbidden)
    monkeypatch.setattr(main_module, "PreflightChecker", forbidden)
    assert main_module.run(["--import-smoke", "--data-root", str(data)]) == 0
    assert imported == [
        "sherpa_onnx",
        "sounddevice",
        "soundfile",
        "PySide6.QtWidgets",
    ]
    assert "VIM2 native import smoke passed." in capsys.readouterr().out
    assert (data / "config" / "settings.json").is_file()
    assert not (tmp_path / "home").exists()


def test_import_smoke_reports_import_failure_without_success(tmp_path, monkeypatch, capsys):
    from vim2 import main as main_module

    bundle = _bundle_with_defaults(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    def fail_import(name):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(main_module.importlib, "import_module", fail_import)
    assert main_module.run(["--import-smoke", "--data-root", str(tmp_path / "data")]) == 1
    captured = capsys.readouterr()
    assert "missing sherpa_onnx" in captured.err
    assert "native import smoke passed" not in captured.out


def _powershell_commands(script: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in script.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped.startswith(",@("):
            continue
        quotes = []
        token = []
        in_quote = False
        for char in stripped:
            if char == "'":
                if in_quote:
                    quotes.append("".join(token))
                    token = []
                in_quote = not in_quote
                continue
            if in_quote:
                token.append(char)
        commands.append(quotes)
    return commands


def test_windows_builder_binds_complete_python_argv_without_array_splat():
    script = (ROOT / "tools" / "build-windows-installer.ps1").read_text(encoding="utf-8")
    assert "Invoke-Python @(" not in script
    assert "function Invoke-Python" in script
    assert "[string[]] $Arguments" in script or "[string[]]$Arguments" in script
    assert "Invoke-Python -Arguments" in script
    assert "if ($LASTEXITCODE -ne 0)" in script
    assert "[switch] $SelfTest" in script
    assert "--import-smoke" in script
    command_lines = [
        line
        for line in script.splitlines()
        if line.split("#", 1)[0].strip().startswith(",@(")
    ]
    assert command_lines
    assert all(not line.lstrip().startswith("#") for line in command_lines)
    assert any("--root" in line and "$Root" in line for line in command_lines)
    commands = _powershell_commands(script)
    assert commands == [
        [
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-windows-cpu.lock",
            "-r",
            "requirements-build.lock",
            "pytest==9.1.1",
        ],
        ["tools/download-cpu-model.py"],
        ["tools/verify-installer-layout.py", "--source"],
        ["-m", "pytest", "-q", "-p", "no:cacheprovider"],
        ["-m", "vim2", "--root", "--check"],
        [
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            "dist/build",
            "--workpath",
            "build/pyinstaller",
            "packaging/vim2.spec",
        ],
        ["tools/verify-installer-layout.py", "dist/build/VIM2"],
    ]
    assert "$Root" in script and "dist/build/VIM2" in script
    workflow = (ROOT / ".github/workflows/build-installers.yml").read_text(encoding="utf-8")
    assert "build-windows-installer.ps1 -SelfTest" in workflow
    assert "tools/verify-installer-layout.py dist/build/VIM2" in workflow
    assert "VIM2-0.1.0-windows-x64-Setup.exe" in workflow
    macos = (ROOT / "tools" / "build-macos-installer.sh").read_text(encoding="utf-8")
    assert "--import-smoke" in macos
    assert "--skip-runtime-check" in macos
    assert "--data-root" in macos
    assert "verify-installer-layout.py --source" in macos
    assert macos.index("--import-smoke") > macos.index("--skip-runtime-check")


def test_windows_argv_self_test_runs_when_pwsh_is_available():
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is not installed; Actions workflow runs the argv self-test")
    completed = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(ROOT / "tools" / "build-windows-installer.ps1"), "-SelfTest"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_docs_separate_installer_layout_from_signing_and_model_mirrors():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "未完成 `.app` 签名" not in readme
    assert "发布布局" not in readme
    assert "Developer ID" in readme and "公证" in readme and "正式验收" in readme
    assert "ModelScope" in readme or "modelscope" in readme
    assert "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25" in readme
    assert "license" in readme.lower() or "许可" in readme
    installation = (ROOT / "INSTALLATION.md").read_text(encoding="utf-8")
    assert "modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx" in installation
    assert "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25" in installation
    assert "68818b2313fe77bd06f6a7c5068ff3ef59d02b8a" in installation
    assert "许可" in installation
    manifest = json.loads((ROOT / "release-manifest.json").read_text(encoding="utf-8"))
    portable = manifest["scopes"]["portable"]
    installer = manifest["scopes"]["installer"]
    assert "GPU" in portable["description"] or "CUDA" in portable["description"]
    assert "qwen3-asr-0.6b-fp16" in portable["models"]
    assert installer["model"] == "qwen3-asr-0.6b-int8-cpu"
    assert "qwen3-asr-0.6b-fp16" not in installer.get("models", [installer["model"]])
    assert "torch" in installer["excluded_from_installer"]
    assert manifest["runtime_scope"] != "installer"
    assert manifest["installers"]["sbom"] == "scopes.installer"
