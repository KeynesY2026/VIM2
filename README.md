# VIM2

VIM2 是面向 Windows 10/11 x64 的本地语音输入工具。它使用官方
Qwen3-ASR 模型在 NVIDIA GPU 上离线识别，并通过系统剪贴板和
`SendInput` 将最终文本粘贴回录音开始时的窗口。

## 运行要求

- Python 3.10-3.13（当前验证版本：Python 3.13）
- Windows 11 x64（Windows 10 x64 为目标支持平台）
- NVIDIA GPU；CUDA 12.8 运行时所需最低驱动版本 570.65
- 已在全局 Python 环境安装 `requirements.lock` 中列出的依赖
- 发布目录中已经准备好的 `.models`

正常启动不联网、不安装依赖，也不使用用户目录中的 Hugging Face 缓存。

## 启动

双击 `Start.cmd`。它通过隐藏的 Windows PowerShell 启动 `pythonw.exe`，
桌面不会保留命令行窗口。`start.bat` 保留为诊断入口，可显示启动错误和
执行 `start.bat --check`。首次运行默认加载 Qwen3-ASR 0.6B FP16。
默认全局热键是右 Alt；再次按下停止录音，录音期间按 Esc 取消。
实时识别默认每秒发起一次预览，间隔可在 `config/settings.json` 中通过
`preview_interval_ms` 调整为 250–1000 毫秒；前一次推理未完成时不会排队。
每次推理最多处理最近 12 秒音频，并在存在
可靠句子锚点时与稳定前缀合并，避免长录音让预览越来越慢。停止后仍会基于
封存音频执行尾段优化或完整识别，不会把临时预览直接当作最终结果。
音频输入 overflow 不再中止整段录音，而是在结果完成后显示警告。错误悬浮
提示会在 10 秒后自动淡出；成功粘贴后悬浮窗立即淡出。

配置位于：

- `config/settings.json`：模型、最长录音时长和实时预览间隔。
- `config/hotkey.conf`：单键或以 `+` 分隔的组合键。

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
RTF 和峰值显存，并执行 `REQUIREMENTS.md` 中的门槛检查。当前需求记录的
1.7B INT8 POC RTF 为 1.284，正式发布前必须在目标 RTX A2000 Laptop 4GB
上重新验证并达到 RTF 小于 1.0，否则属于发布阻塞项。

## 开发验证

```powershell
python -m pytest tests -q
python -m compileall -q app tools tests
```
