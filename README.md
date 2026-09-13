# VIM2

VIM2 是面向 Windows 10/11 x64 的本地语音输入工具。它使用官方
Qwen3-ASR 模型在 NVIDIA GPU 上离线识别，并通过系统剪贴板和
`SendInput` 将最终文本粘贴回录音开始时的窗口。

## 运行要求

- Python 3.10-3.13（当前验证版本：Python 3.13）
- Windows 11 x64（Windows 10 x64 为目标支持平台）
- NVIDIA GPU；CUDA 12.8 运行时所需最低驱动版本 570.65
- 发布目录中已经准备好的 `runtime/site-packages` 和 `.models`

正常启动不联网、不安装依赖，也不使用用户目录中的 Hugging Face 缓存。

## 启动

双击 `start.bat`。首次运行默认加载 Qwen3-ASR 0.6B FP16。默认全局
热键是右 Alt；再次按下停止录音，录音期间按 Esc 取消。

配置位于：

- `config/settings.json`：模型和最长录音时长。
- `config/hotkey.conf`：单键或以 `+` 分隔的组合键。

## 发布准备

在联网的构建机器上执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\prepare-release.ps1
```

该脚本把锁定依赖安装到 `runtime/site-packages`。模型只会从需求文档指定
的本机 Hugging Face snapshot 缓存复制；完整目标目录会跳过，不会覆盖，
也不会移动或删除原缓存。脚本最后生成模型文件 SHA-256 清单
`release-files.sha256.json`。

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
