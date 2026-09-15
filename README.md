# VIM2

VIM2 是面向 Windows 10/11 x64 的本地语音输入工具。它使用
Qwen3-ASR 模型在 NVIDIA GPU 或 CPU 上离线识别，并通过系统剪贴板和
`SendInput` 将最终文本粘贴回录音开始时的窗口。

## 运行要求

- Python 3.10-3.13（当前验证版本：Python 3.13）
- Windows 11 x64（Windows 10 x64 为目标支持平台）
- GPU 模式需要 NVIDIA GPU；CUDA 12.8 运行时所需最低驱动版本 570.65
- CPU 模式不需要 NVIDIA GPU 或 CUDA
- 已在全局 Python 环境安装 `requirements.lock` 中列出的依赖
- 发布目录中已经准备好的 `.models`

正常启动不联网、不安装依赖，也不使用用户目录中的 Hugging Face 缓存。

### macOS Apple Silicon CPU Phase 1

macOS 当前仅提供有界的 CPU MVP：原生 arm64 Python 3.11、
`qwen3-asr-0.6b-int8-cpu` 和现有 sherpa-onnx 0.6B INT8 模型。不会静默改用
CUDA、BitsAndBytes 或 MPS；配置为快速/高精度模型时，平台预检会直接拒绝。
Windows 仍使用原 `requirements.lock`、启动脚本和全部三个模型；macOS 依赖单独
列于 `requirements-macos-cpu.lock`。macOS 托盘中的“粘贴快捷键”子菜单默认选择
“macOS：⌘V”；向本机 Mac 应用粘贴时使用 ⌘V，向 Windows 或远程桌面粘贴时可改选
“Windows / 远程桌面：Ctrl+V”。选择会先以单文件原子替换写入
`settings.json`，成功后立即用于后续粘贴，下次启动恢复；写入失败时配置、运行时和
菜单选择均保持旧值，且不会重写 `hotkey.conf`。只有“就绪”状态可切换；录音、
实时/最终识别、等待重试和模型切换期间菜单禁用。
Windows 本机始终使用 Ctrl+V，不显示此菜单，也不受该 macOS 配置影响。

启用前将 `config/settings.json` 的 `selected_model` 设为
`qwen3-asr-0.6b-int8-cpu`，并把完整模型放在：

```text
.models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/
```

然后从可写的源码目录用原生 Python 3.11 检查并启动：

```bash
PYTHONPATH=app python3.11 -m vim2 --root "$PWD" --check
PYTHONPATH=app python3.11 -m vim2 --root "$PWD"
```

启动预检会明确报告 Rosetta、Python 版本、模型文件、PySide6/sounddevice/
sherpa-onnx/PyObjC 原生模块，以及已拒绝的辅助功能、输入监控或麦克风权限。
应在“系统设置 → 隐私与安全性”中把权限授予稳定的 Python 启动器；麦克风首次
访问仍可能由 macOS 弹出授权请求。

本阶段尚未在真实 M1 上验证模型加载/识别速度与准确率、16 kHz 音频设备行为、
完整 TCC 授权流程、事件注入兼容性、arm64 wheel 安装，也未完成 `.app` 签名、
公证或发布布局。因此不能把本节视为正式 macOS 发布支持声明。

## 启动

双击 `Start.cmd`。它通过隐藏的 Windows PowerShell 启动 `pythonw.exe`，
桌面不会保留命令行窗口。`start.bat` 保留为诊断入口，可显示启动错误和
执行 `start.bat --check`。仓库当前检入的 `config/settings.json` 选择
Qwen3-ASR 0.6B INT8 CPU；Windows 删除配置或省略 `selected_model` 时仍沿用
原有 Qwen3-ASR 0.6B FP16 缺省回退，不改变 Windows 原行为。默认全局热键是
右 Alt；再次按下停止录音，录音期间按 Esc 取消。
实时识别默认每秒发起一次预览，间隔可在 `config/settings.json` 中通过
`preview_interval_ms` 调整为 250–1000 毫秒；前一次推理未完成时只保留一个
最新待处理请求，完成后立即识别最新音频，不处理已经过期的中间快照。
每次推理最多处理最近 12 秒音频，并在存在可靠句子锚点时与稳定前缀合并，
避免长录音让预览越来越慢。多句预览会立即确认最后一个完整句之前的内容；
只有一个完整句时，仍要求连续三次预览一致。

再次按下热键时，程序立即封存录音并取消仍在进行的预览。尚未进入模型的预览
不会执行；已经进入 Transformers 生成阶段的预览会通过停止条件尽快结束，
其部分结果会被丢弃。随后程序基于封存音频执行尾段优化或完整识别，不会把
临时预览直接作为最终结果。尾段在检查点前保留的重叠默认是 5 秒，可通过
`tail_overlap_seconds` 配置为 1–15 秒。
音频输入 overflow 不再中止整段录音，而是在结果完成后显示警告。错误悬浮
提示会在 10 秒后自动淡出；成功粘贴后悬浮窗立即淡出。

配置位于：

- `config/settings.json`：模型、最长录音时长、实时预览间隔、尾段重叠秒数，以及
  macOS 粘贴快捷键。
- `config/hotkey.conf`：单键或以 `+` 分隔的组合键。

当前检入配置（不是字段缺失时的 Windows 缺省回退值）：

```json
{
  "macos_paste_shortcut": "command-v",
  "max_recording_seconds": 90,
  "preview_interval_ms": 1000,
  "selected_model": "qwen3-asr-0.6b-int8-cpu",
  "tail_overlap_seconds": 5
}
```

`selected_model` 的三个有效值为：

- `qwen3-asr-0.6b-int8-cpu`（CPU）
- `qwen3-asr-0.6b-fp16`（FAST/快速）
- `qwen3-asr-1.7b-int8`（ACCURATE/高精度）

Windows 三者均可选择，且字段缺失时仍以 FAST/0.6B FP16 作为原有缺省值；
macOS Phase 1 仅允许 CPU，配置 FAST 或 ACCURATE 会在预检中明确失败。
`macos_paste_shortcut` 的有效值为 `command-v`（缺失时默认，本机 Mac）和
`control-v`（Windows / 远程桌面）；其他值会在启动时作为无效配置明确拒绝。
该字段不改变 Windows 本机固定的 Ctrl+V 粘贴行为。

CPU 模式使用
`.models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` 中的预量化 ONNX
模型和 `sherpa-onnx` CPU provider。它是独立推理后端，不加载 Torch、
Qwen-ASR、BitsAndBytes 或 CUDA，不改变原有两个 GPU 模式。首次使用前需
安装锁定的 `sherpa-onnx==1.13.8`；该模型来自第三方 ONNX 转换，并非 Qwen
官方发布的预量化 checkpoint。

## 发布准备

在联网的构建机器上执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\prepare-release.ps1
```

该脚本只检查全局 Python 版本和依赖，不创建虚拟环境，也不在项目目录安装
Python 包。模型只会从需求文档指定的本机 Hugging Face snapshot 缓存复制；
完整目标目录会跳过，不会覆盖，也不会移动或删除原缓存。脚本最后生成模型
文件 SHA-256 清单 `release-files.sha256.json`。

如全局依赖尚未安装，可由用户显式执行：

```powershell
python -m pip install -r .\requirements.lock `
  --extra-index-url https://download.pytorch.org/whl/cu128
```

该安装不是 `Start.cmd`、`start.bat` 或应用启动流程的一部分。

## A/B 性能测试

每个模型使用独立进程，先预热，再对每条音频运行三次并取中位数：

```powershell
python .\tools\benchmark.py .\dataset\a.wav .\dataset\b.wav `
  --model both --runs 3 --output .\benchmark-result.json
```

报告包含模型加载时间、纯推理时间、RTF、每 100 ms 采样的进程峰值显存
和全部原始文本。

## 准确率验收

复制并扩充 `tools/acceptance-dataset.example.json`。正式数据集必须冻结版本，
包含至少 100 条音频及人工校对文本：

```powershell
python .\tools\acceptance.py .\dataset\manifest.json `
  --runs 3 --output .\acceptance-result.json
```

报告分别计算中文 CER、英文 WER、中英文边界错误句数、专有名词完全正确率、
RTF 和峰值显存，并执行 `REQUIREMENTS.md` 中的门槛检查。

当前带背景音乐的 44.745 秒 POC 基线：

| 模型 | 整段推理 | RTF | 峰值显存 |
|---|---:|---:|---:|
| 0.6B INT8 CPU | 23.98 秒 | 0.536 | 不适用 |
| 0.6B FP16 | 12.51 秒 | 0.280 | 1.80 GiB |
| 1.7B INT8 | 49.58 秒 | 1.108 | 2.81 GiB |

以上结果于 2026-09-14 在同一进程隔离方案下取得，每个模型先预热，再运行
三次并取中位数。CPU INT8 无需 GPU 且冷加载最快，但该机器上的推理速度约为
0.6B FP16 的 52%；它应作为无 CUDA 回退模式，而不是速度优先模式。
1.7B 对真实尾段的历史耗时为：4 秒音频 4.22 秒、8 秒音频 6.77 秒、12 秒
音频 9.82 秒。因此 0.6B FP16 仍是 Windows 字段缺失时的缺省主模型；仓库
当前检入配置选择 CPU，1.7B 保留为 Windows 用户可选的高精度模式。

## 开发验证

```powershell
python -m pytest tests -q
python -m compileall -q app tools tests
```
