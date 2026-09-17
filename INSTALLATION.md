# VIM2 Windows 安装手册

本手册面向第一次安装 Python 程序的用户，请按顺序操作。

> 当前版本支持 Windows 10/11 x64。macOS 支持正在路上。

## 1. 选择模型

第一次只需安装一个模型：

| 模型 | 需要 NVIDIA 显卡 | 放置目录 | 建议 |
|---|---|---|---|
| 0.6B INT8 CPU | 否 | `.models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | 默认，无 NVIDIA 显卡 |
| 0.6B FP16 | 是 | `.models\Qwen3-ASR-0.6B` | 速度和准确率均衡 |
| 1.7B INT8 | 是 | `.models\Qwen3-ASR-1.7B-INT8` | 可选高精度，速度较慢 |

有 NVIDIA 显卡（2GB+）时推荐默认 0.6B FP16，否则选择 CPU 模型。VIM2 启动时不会自动下载模型，目录名称和层级必须与上表一致。

## 2. 安装 Python 3.13 x64

VIM2 支持 Python 3.10 至 3.13，当前验证版本为 Python 3.13。不要安装 Python 3.14 或 32 位版本。

1. 打开 [Python Windows 下载页](https://www.python.org/downloads/windows/)。
2. 找到最新的 Python 3.13 稳定版。
3. 下载 **Windows installer (64-bit)**。
4. 运行安装程序，勾选 **Add python.exe to PATH**。
5. 点击 **Install Now**。
6. 安装完成后重新打开 PowerShell。

执行以下命令验证：

```powershell
python --version
python -m pip --version
```

第一行应显示 `Python 3.13.x`。如果提示找不到 `python`，重新运行安装程序，选择 **Modify**，确认 Python 已加入 PATH。

## 3. GPU 模式安装 NVIDIA 驱动

CPU 模式可跳过本节。GPU 模式需要 NVIDIA 显卡，驱动版本不得低于 `570.65`。

```powershell
nvidia-smi
```

确认输出中有显卡名称，且 `Driver Version` 不低于 `570.65`。如果命令不存在或版本过低：

1. 打开 [NVIDIA 驱动下载页](https://www.nvidia.com/Download/index.aspx)。
2. 选择自己的显卡型号和 Windows 版本。
3. 下载并安装最新驱动。
4. 重启电脑，再执行 `nvidia-smi`。

不需要单独安装 CUDA Toolkit、cuDNN、FFmpeg、.NET 或 Visual Studio。

## 4. 准备 VIM2 文件夹

1. 获取 VIM2 发布包或项目目录。
2. 如果是 ZIP，右键选择 **全部解压缩**，不要直接在压缩包中运行。
3. 将文件夹放在固定位置，例如 `D:\Apps\VIM2`。
4. 确认根目录有 `Start.cmd`、`start.bat`、`requirements.lock`、`app` 和 `config`。
5. 在资源管理器中打开 VIM2 根目录，点击地址栏，输入 `powershell` 后按 Enter。

后续命令都在这个 PowerShell 窗口中执行。

## 5. 安装 Python 依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
```

第二条命令会下载较多文件。完成后检查基础依赖：

```powershell
python -c "import PySide6, numpy, sounddevice, soundfile; print('基础依赖正常')"
```

GPU 模式再检查 CUDA：

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
```

GPU 模式必须显示 `CUDA 可用: True`。如果显示 `False`，请更新 NVIDIA 驱动并重启。

## 6. 下载并放置模型

先创建模型总目录：

```powershell
New-Item -ItemType Directory -Force .\.models | Out-Null
```

只执行所选模型对应的小节。

### B. 默认 0.6B INT8 CPU 模型

这是第三方转换的 ONNX INT8 模型，并非 Qwen 官方预量化 checkpoint。来源：[zengshuishui/Qwen3-ASR-onnx](https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/files)。

1. 创建模型目录：

```powershell
New-Item -ItemType Directory -Force .\.models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25\tokenizer | Out-Null
```

2. 在 ModelScope 页面进入 `model_0.6B`，下载 `conv_frontend.onnx`、`encoder.int8.onnx` 和 `decoder.int8.onnx`。
3. 将三个文件放入 `.models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25\`。
4. 在页面中进入 `tokenizer`，下载 `merges.txt`、`tokenizer_config.json` 和 `vocab.json`。
5. 将三个文件放入上述模型目录的 `tokenizer\` 文件夹。
6. 用记事本打开 `config\settings.json`，将模型改为：

```json
"selected_model": "qwen3-asr-0.6b-int8-cpu"
```

只修改该值，保留逗号、引号和其他设置。

### A. 0.6B FP16 GPU 模型

来源：[Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)。

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-ASR-0.6B', revision='5eb144179a02acc5e5ba31e748d22b0cf3e303b0', local_dir=r'.models\Qwen3-ASR-0.6B')"
```

下载后应直接存在：

```text
VIM2\.models\Qwen3-ASR-0.6B\config.json
VIM2\.models\Qwen3-ASR-0.6B\preprocessor_config.json
VIM2\.models\Qwen3-ASR-0.6B\tokenizer_config.json
VIM2\.models\Qwen3-ASR-0.6B\model.safetensors
```

不要形成 `Qwen3-ASR-0.6B\Qwen3-ASR-0.6B\config.json` 这样的双层目录。默认配置已经选择模型 ID `qwen3-asr-0.6b-fp16`，无需修改。

### C. 可选 1.7B INT8 GPU 模型

来源：[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)。下载的是官方权重，VIM2 加载时通过 BitsAndBytes 使用 INT8。

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-ASR-1.7B', revision='7278e1e70fe206f11671096ffdd38061171dd6e5', local_dir=r'.models\Qwen3-ASR-1.7B-INT8')"
```

确认模型文件直接位于 `.models\Qwen3-ASR-1.7B-INT8\`。要首次启动就使用它，将 `config\settings.json` 中的模型改为：

```json
"selected_model": "qwen3-asr-1.7b-int8"
```

也可以启动后从托盘的“识别模型”菜单切换。

## 7. 检查安装

确保配置中选择的模型已经下载，然后执行：

```powershell
.\start.bat --check
```

成功时显示 `VIM2 preflight check passed.`。失败时会列出缺少的依赖或模型文件，按提示修复后重新检查。

## 8. 首次启动

1. 双击 `Start.cmd`。
2. 等待托盘中的 VIM2 图标变为绿色。
3. 打开记事本并点击输入位置。
4. 按一次右 Alt 开始录音，说话后再按一次右 Alt。
5. 等待识别，文字会自动粘贴到记事本。

录音期间按 Esc 可取消。如果 Windows 询问麦克风权限，请允许桌面应用访问麦克风。以后只需双击 `Start.cmd`。

## 9. 常见问题

- **双击后没有反应：** 运行 `start.bat` 查看错误，或打开 `runtime\vim2.log`。
- **Model is incomplete：** 模型目录、层级不正确或下载未完成；重点检查是否多套一层同名目录。
- **CUDA unavailable：** 执行 `nvidia-smi` 和 `python -c "import torch; print(torch.cuda.is_available())"`，更新驱动并重启；无 NVIDIA 显卡请改用 CPU 模型。
- **Hugging Face 下载失败：** 重新运行同一命令会复用已完成文件。也可在另一台联网电脑下载完整模型目录后复制到 `.models`。
- **CPU 模型仍报 GPU 错误：** 确认配置中的 `selected_model` 是 `qwen3-asr-0.6b-int8-cpu`。
- **录音没有声音：** 在 Windows 的 **设置 > 系统 > 声音 > 输入** 中选择默认麦克风，并允许桌面应用访问。
