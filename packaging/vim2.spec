# -*- mode: python ; coding: utf-8 -*-
"""Only named default configs and the six pinned CPU model files enter the bundle."""
from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs

repo = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(repo / 'app'))
from vim2.models import MODEL_SPECS, ModelId, _REQUIRED_SHERPA_FILES
model_name = MODEL_SPECS[ModelId.CPU].directory_name
model = repo / '.models' / model_name
config = repo / 'config'
# Explicit input files only. No recursive repository or user data copy.
datas = []
for filename in ('settings.json', 'hotkey.conf'):
    source = config / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    datas.append((str(source), 'config'))
# Do not embed the ignored local hotwords.txt (it may contain private data).
source = repo / 'packaging' / 'hotwords.template.txt'
if not source.is_file():
    raise FileNotFoundError(source)
datas.append((str(source), 'config'))
for filename in _REQUIRED_SHERPA_FILES:
    source = model / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    datas.append((str(source), str(Path('.models') / model_name / Path(filename).parent)))

# The module graph and PyInstaller soundfile/sounddevice hooks collect Python
# modules and their native libraries; sherpa additionally ships adjacent dylibs.
binaries = collect_dynamic_libs('sherpa_onnx')
hiddenimports = ['sherpa_onnx', 'sounddevice', 'soundfile', 'numpy']
if sys.platform == 'darwin':
    # Dynamic imports in native adapter are not visible to modulegraph.
    hiddenimports += ['objc', 'AppKit', 'Quartz', 'ApplicationServices',
                      'AVFoundation', 'CoreFoundation', 'Foundation']

analysis = Analysis(
    [str(repo / 'packaging' / 'entry.py')],
    pathex=[str(repo / 'app')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'qwen_asr', 'transformers', 'bitsandbytes',
              'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.QtWebEngineQuick', 'PySide6.QtMultimedia',
              'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuickWidgets'],
    noarchive=False,
)
# Built-in hooks may include C/C++ headers, README and CLI source assets.
# They are not runtime files; keep dependencies, native binaries and config.
analysis.datas = [entry for entry in analysis.datas
                  if not entry[0].lower().endswith(('.h', '.hpp', '.md', '.py'))]
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True,
    name='VIM2', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    target_arch='arm64' if sys.platform == 'darwin' else None,
)
coll = COLLECT(exe, analysis.binaries, analysis.datas,
               strip=False, upx=False, name='VIM2')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='VIM2.app', icon=None,
                 bundle_identifier='com.vim2.desktop',
                 info_plist={'CFBundleShortVersionString': '0.1.0',
                             'CFBundleVersion': '0.1.0',
                             'NSMicrophoneUsageDescription': 'VIM2 needs microphone access for offline transcription.'})
