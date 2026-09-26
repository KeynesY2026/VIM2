# VIM2

**按一下说话，再按一下，文字就出现在光标处。**

VIM2 是以 Windows 10/11 x64 为主、并逐步支持 macOS Apple Silicon 的本地语音
输入工具。它能在记事本、浏览器、聊天软件、IDE 和远程桌面中工作。只需在自己的
电脑安装一次，连接不同远程机器时无需重复安装模型。语音识别完全在本机完成，音频
和文字不上传，也没有云服务费用。

> Windows 10/11 x64 是当前正式支持平台；macOS Apple Silicon 当前为有界的 CPU
> Phase 1 MVP，限制和未验证项见下文。

[第一次使用：查看完整安装手册](INSTALLATION.md)

## 为什么选择 VIM2

- **真正离线：** 模型在本机运行，断网也能识别，语音不离开电脑。
- **一次安装，处处可用：** 连接多少台远程机器都不必重复部署，直接向当前远程桌面输入文字。
- **哪里都能输入：** 不局限于单个编辑器，最终文字直接粘贴到原来的光标处。
- **边说边看：** 录音过程中实时显示预览，停止后再生成可靠的最终文本。
- **中英混合：** 支持中文、英文、中英文混说以及多种中文方言。
- **硬件可选：** 有 NVIDIA GPU 可追求速度或精度，没有独显也能使用 CPU 模型。

## 运行要求

- Python 3.10–3.13；macOS Phase 1 固定使用原生 arm64 Python 3.11。
- Windows 11 x64（Windows 10 x64 为目标支持平台），或满足下述限制的 macOS
  Apple Silicon。
- GPU 模式需要 NVIDIA GPU；CUDA 12.8 运行时所需最低驱动版本为 570.65。
- CPU 模式不需要 NVIDIA GPU 或 CUDA。
- 已安装对应锁定文件中的依赖，发布目录中已经准备好 `.models`。

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
实时/最终识别、等待重试和模型切换期间菜单禁用。Windows 本机始终使用 Ctrl+V，
不显示此菜单，也不受该 macOS 配置影响。

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

Windows 完成安装后，双击 `Start.cmd`。它通过隐藏的 Windows PowerShell 启动
`pythonw.exe`，桌面不会保留命令行窗口。等待任务栏右下角的 VIM2 托盘图标变为
绿色。双击后没有反应时，查看 `runtime/vim2.log`。安装检查命令为：

```powershell
$env:PYTHONPATH = "$PWD\app"
python -m vim2 --check
```

仓库当前检入的 `config/settings.json` 选择 Qwen3-ASR 0.6B INT8 CPU；Windows
删除配置或省略 `selected_model` 时仍沿用 Qwen3-ASR 0.6B FP16 缺省回退，不改变
Windows 原行为。默认全局热键是右 Alt；再次按下停止录音，录音期间按 Esc 取消。

实时识别默认每秒发起一次预览，间隔可在 `config/settings.json` 中通过
`preview_interval_ms` 调整为 250–1000 毫秒；前一次推理未完成时不排队，完成后
等待下一次定时请求，不立即连续补跑。`preview_window_seconds` 控制每次只复制并
识别最近的音频（默认 8 秒，可设为 1–30 秒），在存在可靠句子锚点时与稳定前缀
合并，避免长录音让预览越来越慢。
多句预览会立即确认最后一个完整句之前的内容；只有一个完整句时，仍要求连续三次
预览一致。

再次按下热键时，程序立即封存录音并取消仍在进行的预览。尚未进入模型的预览不会
执行；已经进入 Transformers 生成阶段的预览会通过停止条件尽快结束，其部分结果会
被丢弃。随后程序基于封存音频执行尾段优化或完整识别，不会把临时预览直接作为最终
结果。尾段在检查点前保留的重叠默认是 5 秒，可通过 `tail_overlap_seconds` 配置为
1–15 秒。音频输入 overflow 不再中止整段录音，而是在结果完成后显示警告。错误悬浮
提示会在 10 秒后自动淡出；成功粘贴后悬浮窗立即淡出。

## 使用

1. 点击任意应用中要输入文字的位置。
2. 按一次右 Alt 开始录音。
3. 再按一次右 Alt 停止；识别完成后文字会自动粘贴。
4. 在本地应用中，录音期间按 Esc 可取消，不会识别或粘贴文字。

右键单击托盘图标可以切换模型、编辑或重新加载热词、在 macOS 上选择粘贴快捷键，
或退出。

远程使用时，VIM2 仍然运行在自己的电脑上；请确认远程桌面或远控软件已允许剪贴板
同步。远程机器不需要安装 VIM2、Python 或模型。部分 RDP 客户端会直接将 Esc 转发
给远程系统，本机 VIM2 无法捕获，因此不支持在 RDP 窗口内使用 Esc 取消录音。

## 模型

| 模型 | 设备 | 特点 |
|---|---|---|
| Qwen3-ASR 0.6B INT8 | CPU | 当前检入配置的默认选择，无需 NVIDIA 显卡 |
| Qwen3-ASR 0.6B FP16 | NVIDIA GPU | Windows 配置字段缺失时的回退值，速度和准确率均衡 |
| Qwen3-ASR 1.7B INT8 | NVIDIA GPU | 更高精度，处理速度较慢 |

程序一次只加载一个模型。可在托盘的“识别模型”菜单中切换，选择会自动保存。
Windows 三者均可选择；macOS Phase 1 仅允许 CPU，配置 FAST 或 ACCURATE 会在
预检中明确失败。

CPU 模式使用
`.models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` 中的预量化 ONNX
模型和 `sherpa-onnx` CPU provider。它是独立推理后端，不加载 Torch、
Qwen-ASR、BitsAndBytes 或 CUDA，不改变原有两个 GPU 模式。首次使用前需安装锁定
的 `sherpa-onnx==1.13.8`；该模型来自第三方 ONNX 转换，并非 Qwen 官方发布的
预量化 checkpoint。

## 热词与配置

在 `config/hotwords.txt` 中每行填写一个人名、产品名或专业术语。Windows 上从
托盘选择“打开热词文件”使用记事本编辑；关闭记事本进程后，VIM2 比较打开前后
的文件修改时间，仅在发生变化时自动加载新热词。编辑期间不会刷新；如果关闭时
正在录音或识别，会在恢复就绪后加载。macOS 上此入口使用系统默认关联程序打开，
编辑完成后请从托盘选择“重新加载热词”；该手动入口在 Windows 上也可使用。
热词会提高相关词出现的概率，但不会强制替换识别结果。

- `config/settings.json`：模型、最长录音时间、实时预览（`preview_window_seconds`
  默认 8 秒）、尾段重叠（`tail_overlap_seconds` 默认 5 秒）、数字规范化以及
  macOS 粘贴快捷键。预览窗口与稳定句子边界前的音频重叠用途不同。
- `config/hotkey.conf`：全局热键，默认是 `RightAlt`；支持单键或以 `+` 分隔的
  组合键。
- `config/hotwords.txt`：一行一个热词或短语。

当前检入配置（不是字段缺失时的 Windows 缺省回退值）：

```json
{
  "macos_paste_shortcut": "command-v",
  "max_recording_seconds": 300,
  "normalize_numbers": true,
  "preview_interval_ms": 1000,
  "preview_window_seconds": 8,
  "selected_model": "qwen3-asr-0.6b-int8-cpu",
  "tail_overlap_seconds": 5
}
```

`selected_model` 的三个有效值为：

- `qwen3-asr-0.6b-int8-cpu`（CPU）
- `qwen3-asr-0.6b-fp16`（FAST/快速）
- `qwen3-asr-1.7b-int8`（ACCURATE/高精度）

`macos_paste_shortcut` 的有效值为 `command-v`（缺失时默认，本机 Mac）和
`control-v`（Windows / 远程桌面）；其他值会在启动时作为无效配置明确拒绝。若
`~/.vim2/settings.local.json` 覆盖该字段，托盘切换会拒绝并提示先删除该本地覆盖，
避免重启后恢复旧值。该字段不改变 Windows 本机固定的 Ctrl+V 粘贴行为。`normalize_numbers` 默认为
`true`，可单独关闭中英文口述数字规范化。默认最长录音时间为 300 秒。

## 发布准备

在联网的构建机器上执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\prepare-release.ps1
```

该脚本只检查全局 Python 版本和依赖，不创建虚拟环境，也不在项目目录安装 Python
包。模型只会从需求文档指定的本机 Hugging Face snapshot 缓存复制；完整目标目录会
跳过，不会覆盖，也不会移动或删除原缓存。脚本最后生成模型文件 SHA-256 清单
`release-files.sha256.json`。

如全局依赖尚未安装，可由用户显式执行：

```powershell
python -m pip install -r .\requirements.lock `
  --extra-index-url https://download.pytorch.org/whl/cu128
```

该安装不是 `Start.cmd` 或应用启动流程的一部分。

## A/B 性能测试

每个模型使用独立进程，先预热，再对每条音频运行三次并取中位数：

```powershell
python .\tools\benchmark.py .\dataset\a.wav .\dataset\b.wav `
  --model both --runs 3 --output .\benchmark-result.json
```

报告包含模型加载时间、纯推理时间、RTF、每 100 ms 采样的进程峰值显存和全部原始
文本。

## 准确率验收

复制并扩充 `tools/acceptance-dataset.example.json`。正式数据集必须冻结版本，
包含至少 100 条音频及人工校对文本：

```powershell
python .\tools\acceptance.py .\dataset\manifest.json `
  --runs 3 --output .\acceptance-result.json
```

报告分别计算中文 CER、英文 WER、中英文边界错误句数、专有名词完全正确率、RTF
和峰值显存，并执行 `REQUIREMENTS.md` 中的门槛检查。

当前带背景音乐的 44.745 秒 POC 基线：

| 模型 | 整段推理 | RTF | 峰值显存 |
|---|---:|---:|---:|
| 0.6B INT8 CPU | 23.98 秒 | 0.536 | 不适用 |
| 0.6B FP16 | 12.51 秒 | 0.280 | 1.80 GiB |
| 1.7B INT8 | 49.58 秒 | 1.108 | 2.81 GiB |

以上结果于 2026-09-14 在同一进程隔离方案下取得，每个模型先预热，再运行三次并取
中位数。CPU INT8 无需 GPU 且冷加载最快，但该机器上的推理速度约为 0.6B FP16 的
52%；它应作为无 CUDA 回退模式，而不是速度优先模式。1.7B 对真实尾段的历史耗时
为：4 秒音频 4.22 秒、8 秒音频 6.77 秒、12 秒音频 9.82 秒。因此 0.6B FP16
仍是 Windows 字段缺失时的缺省主模型；仓库当前检入配置选择 CPU，1.7B 保留为
Windows 用户可选的高精度模式。

## 开发验证

产品要求见 [REQUIREMENTS.md](REQUIREMENTS.md)。验证代码：

```powershell
python -m pytest tests -q
python -m compileall -q app tools tests
```
