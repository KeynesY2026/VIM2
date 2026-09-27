# VIM2 0.1.0 安装版内测 / 源码 Portable 手册

## 完全自包含安装版（未签名内测）

macOS Apple Silicon 原生 arm64：`VIM2-0.1.0-macos-arm64.dmg`，打开后将
`VIM2.app` 拖到 Applications 链接；内含原生 Python、PySide6、sherpa-onnx、
依赖和**仅** `qwen3-asr-0.6b-int8-cpu` 模型。不必安装 Python 或下载模型。
该 app 为 PyInstaller ad-hoc 签名，**没有 Developer ID 签名、没有公证**；
Gatekeeper 可能阻止首次启动，内测人员应核对来源和 SHA-256 后通过系统设置
“隐私与安全性 → 仍要打开”有意识地授权，不能以此替代正式签名与公证。
辅助功能、输入监控、麦克风权限须为实际安装位置的 app 单独授予。

打开 DMG 后先阅读镜像根目录的 `安装说明.txt`。必须把 `VIM2.app`
拖到 Applications 并等待约 1.1GB 复制完成，弹出镜像，再从应用程序里右键打开。
不要在磁盘镜像内双击；包括 Gatekeeper 将镜像内 app 暂时转移到
`/private/var/folders/.../AppTranslocation/<token>/d/...` 的情形，都会显示安装提示
并退出，不会在镜像或临时身份上请求权限。
安装后的首次普通启动会通过系统 API 请求辅助功能和输入监控，并显示说明；
麦克风仍由系统在录音时请求。`--check` / `--import-smoke` 不弹权限请求。
Gatekeeper“仍要打开”和 `xattr` 只是受控内测的最后手段，安装包不会自动执行。

Windows 10/11 x64 的 GitHub Actions / Setup.exe 构建已按用户要求暂停、放弃本轮
打包和重跑；**本轮没有 Windows 安装产物**，不能从 macOS 构建推断 Windows
已完成或已验证。Windows workflow 已移除自动 `push` 触发：向 macOS 分支 push
不会自动运行 Windows job；仅保留 `workflow_dispatch` 作为未来显式人工恢复入口，
本轮不触发。此前设计中的未签名当前用户安装器和 SmartScreen 提示仅是
未来恢复构建时的规划，不是本轮可安装文件。

两端模型只读，用户配置/热词/日志/锁/临时音频均与安装目录分离：
macOS `~/Library/Application Support/VIM2/{config,runtime,temp}`；Windows
`%LOCALAPPDATA%\\VIM2\\{config,runtime,temp}`。首次启动仅补齐缺失的
`settings.json`、`hotkey.conf`、`hotwords.txt`，不覆盖已有数据；安装包仅含通用
`hotwords.template.txt`，绝不打包源码目录内被忽略的个人热词文件；卸载程序
不自动删除用户目录。诊断检查可用 `--check --skip-runtime-check --data-root
<临时空目录>` 隔离用户数据，**不启动 GUI 或注入按键**，不等同真机验收。
源码与 `--root` Portable 仍采用旧的仓库根目录布局；下文只适用于源码模式，
不适用于安装版。

**对外分发阻断条件：** CPU 模型系第三方转换版本，当前 Hugging Face model
card 缺失 license metadata，必须独立核实来源、上游及转换权利和许可证；
还需完成 macOS 真机 TCC/粘贴及 Windows 原生/远程桌面验收、发行签名/公证评估。
安装版构建只接受固定镜像
`csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` revision
`68818b2313fe77bd06f6a7c5068ff3ef59d02b8a` 的六个文件，并按
`release-files.sha256.json` 校验。下文源码 Portable 的 ModelScope 页面
`https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx` 只是人工下载入口，
不作为安装版输入；两处都是第三方转换，许可尚未在 HF card 中标明。
首次写入用户文件时先写临时文件并 fsync，再原子替换；失败不保留半截文件，可重试。

## 以下为源码 Portable 安装说明（不是安装版依赖）

本手册下文面向第一次安装 Python 源码的用户，请按顺序操作。

## 1. 选择模型

第一次只需安装一个模型：

| 模型 | 需要 NVIDIA 显卡 | 放置目录 | 建议 |
|---|---|---|---|
| 0.6B INT8 CPU | 否 | `.models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | 默认，无 NVIDIA 显卡 |
| 0.6B FP16 | 是 | `.models\Qwen3-ASR-0.6B` | 速度和准确率均衡 |
| 1.7B INT8 | 是 | `.models\Qwen3-ASR-1.7B-INT8` | 可选高精度，速度较慢 |

默认使用 0.6B INT8 CPU 模型，无需 NVIDIA 显卡；如果有 NVIDIA 显卡（2GB+），也可选用 0.6B FP16 模型以获得更快的速度。VIM2 启动时不会自动下载模型，目录名称和层级必须与上表一致。

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
4. 确认根目录有 `Start.cmd`、`start.ps1`、`requirements.lock`、`app` 和 `config`。
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

### A. 默认 0.6B INT8 CPU 模型

这是第三方转换的 ONNX INT8 模型，并非 Qwen 官方预量化 checkpoint。源码
Portable 可从 ModelScope 人工下载：[zengshuishui/Qwen3-ASR-onnx](https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/files)。
安装版不从该页面下载，只使用固定 revision
`68818b2313fe77bd06f6a7c5068ff3ef59d02b8a` 的 Hugging Face 镜像
`csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`，并核对
`release-files.sha256.json`。两处文件均须独立核实来源与许可证。

1. 创建模型目录：

```powershell
New-Item -ItemType Directory -Force .\.models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25\tokenizer | Out-Null
```

2. 在 ModelScope 页面进入 `model_0.6B`，下载 `conv_frontend.onnx`、`encoder.int8.onnx` 和 `decoder.int8.onnx`。
3. 将三个文件放入 `.models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25\`。
4. 在页面中进入 `tokenizer`，下载 `merges.txt`、`tokenizer_config.json` 和 `vocab.json`。
5. 将三个文件放入上述模型目录的 `tokenizer\` 文件夹。
6. 默认配置已经选择该模型（`"selected_model": "qwen3-asr-0.6b-int8-cpu"`），无需修改 `config\settings.json`。

### B. 0.6B FP16 GPU 模型

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

不要形成 `Qwen3-ASR-0.6B\Qwen3-ASR-0.6B\config.json` 这样的双层目录。要首次启动就使用它，将 `config\settings.json` 中的模型改为：

```json
"selected_model": "qwen3-asr-0.6b-fp16"
```

也可以启动后从托盘的“识别模型”菜单切换。

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
$env:PYTHONPATH = "$PWD\app"
python -m vim2 --check
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

- **双击后没有反应：** 打开 `runtime\vim2.log`，或重新执行“检查安装”中的命令查看错误。
- **Model is incomplete：** 模型目录、层级不正确或下载未完成；重点检查是否多套一层同名目录。
- **CUDA unavailable：** 执行 `nvidia-smi` 和 `python -c "import torch; print(torch.cuda.is_available())"`，更新驱动并重启；无 NVIDIA 显卡请改用 CPU 模型。
- **Hugging Face 下载失败：** 重新运行同一命令会复用已完成文件。也可在另一台联网电脑下载完整模型目录后复制到 `.models`。
- **CPU 模型仍报 GPU 错误：** 确认配置中的 `selected_model` 是 `qwen3-asr-0.6b-int8-cpu`。
- **录音没有声音：** 在 Windows 的 **设置 > 系统 > 声音 > 输入** 中选择默认麦克风，并允许桌面应用访问。
