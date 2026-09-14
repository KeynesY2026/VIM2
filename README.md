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

## 启动

双击 `Start.cmd`。它通过隐藏的 Windows PowerShell 启动 `pythonw.exe`，
桌面不会保留命令行窗口。`start.bat` 保留为诊断入口，可显示启动错误和
执行 `start.bat --check`。首次运行默认加载 Qwen3-ASR 0.6B FP16。
默认全局热键是右 Alt；再次按下停止录音，录音期间按 Esc 取消。
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

- `config/settings.json`：模型、最长录音时长、实时预览间隔和尾段重叠秒数。
- `config/hotkey.conf`：单键或以 `+` 分隔的组合键。
- `config/hotwords.txt`：一行一个热词或短语，用于原生上下文偏置。

可从托盘菜单选择“打开热词文件”进行编辑，再选择“重新加载热词”应用更改。
模型启动和切换时也会自动加载该文件。GPU 模式刷新只替换内存中的不可变
context snapshot，不重新加载权重；CPU 模式需要重建 sherpa-onnx recognizer，
在当前目标机器上约需 4 秒。录音、识别或等待重试期间不能刷新，因此同一录音
的实时预览、最终识别和失败重试始终使用同一份 snapshot。

文件使用 UTF-8，可带 BOM。每行会先去除首尾空白，空行和以 `#` 开头的行会被
忽略，完全相同的重复项只保留第一次出现的位置。最多允许 100 项，每项最多
100 个字符。ASCII 逗号是 CPU 后端的分隔符，因此包含逗号的项目会被明确拒绝；
刷新失败时程序继续使用上一次成功加载的词表。热词只提供 soft bias，不保证
识别结果一定包含指定文字，也不会执行字符串后处理替换。

当前默认配置：

```json
{
  "max_recording_seconds": 90,
  "preview_interval_ms": 1000,
  "selected_model": "qwen3-asr-0.6b-fp16",
  "tail_overlap_seconds": 5
}
```

`selected_model` 也可设为 `qwen3-asr-0.6b-int8-cpu`（CPU）或
`qwen3-asr-1.7b-int8`（高精度）。默认仍为 0.6B FP16；1.7B 不参与默认
实时链路。

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
音频 9.82 秒。因此
0.6B FP16 是默认主模型；1.7B 保留为用户可选的高精度模式。

## 开发验证

```powershell
python -m pytest tests -q
python -m compileall -q app tools tests
```
