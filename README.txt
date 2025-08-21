# Chicken-Mamba

基于深度学习的图像分割和重量预测系统，结合改进的VMUNet分割网络和ConvGRU预测网络，提供传统机器学习方法作为对比。

## 📋 目录

- [功能特性](#功能特性)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [网络架构](#网络架构)
- [环境要求](#环境要求)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## ✨ 功能特性

- 🖼️ **图像分割**: 使用改进的VMUNet进行图像分割
- 🧠 **重量预测**: 使用ConvGRU网络进行重量预测
- 🔄 **方法对比**: 深度学习(ConvGRU) vs 传统机器学习特征提取
- 📦 **批量处理**: 支持批量处理多个样本
- 🖥️ **可视化界面**: 友好的PyQt5图形界面
- 🔧 **多版本支持**: 提供三个独立的GUI版本
- 📤 **模型导出**: 支持将深度学习模型导出为joblib格式

## 🚀 安装说明

### 环境要求

- Python 3.7+
- CUDA 11.0+ (可选，用于GPU加速)
- 内存: 8GB+
- 操作系统: Windows/Linux/macOS

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/chicken-Mamba.git
cd chicken-Mamba
```

2. **创建虚拟环境** (推荐)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

如果没有requirements.txt文件，手动安装核心依赖：
```bash
pip install torch torchvision
pip install opencv-python
pip install PyQt5
pip install numpy pandas matplotlib
pip install scikit-learn joblib
pip install Pillow
pip install einops timm
```

## 🏃‍♂️ 快速开始

### 1. 图像分割
```bash
python vmunet.py
```

### 2. 重量预测
```bash
python gru.py
```

### 3. GUI界面

#### 🎯 选择适合的GUI版本

**传统机器学习方法**（推荐新手）：
```bash
cd soft_unet_use_gui
python start.py
```

**深度学习ConvGRU方法**（推荐进阶用户）：
```bash
cd soft_unet_use_gui_convgru
python start.py
```

**最新VSSM+ConvGRU方法**（推荐研究用户）：
```bash
cd soft_unet_use_gui_vmamba
python start.py
# 或使用专用启动文件
python start_vmamba.py
```

#### 📋 GUI快速使用流程

1. **启动程序** → 选择对应版本启动
2. **加载模型** → 选择分割模型(.pth) + 预测模型(.pth/.joblib)
3. **设置路径** → 选择输入图片文件夹 + 输出结果文件夹
4. **开始预测** → 点击"begin"按钮，等待完成
5. **导出结果** → 点击"Export results"，保存CSV文件

#### 🔄 模型兼容性

| GUI版本 | 分割模型 | 预测模型 | 输出格式 |
|---------|----------|----------|----------|
| 原始版本 | UNet (.pth) | ML (.joblib) | CSV |
| ConvGRU版本 | UNet (.pth) | ConvGRU (.pth) | CSV |
| VSSM版本 | VSSM (.pth) | ConvGRU (.pth) | CSV |

**注意**：所有版本都支持将深度学习模型导出为joblib格式，便于与传统ML方法集成！

## 📖 使用方法

### 命令行使用

#### 图像分割
```python
# 使用改进的VMUNet网络进行图像分割
python vmunet.py
```

#### 重量预测
```python
# 使用ConvGRU网络进行重量预测
python gru.py
```

### GUI界面使用

本项目提供三个独立的GUI版本，每个版本都支持不同的模型组合：

#### 🖥️ GUI版本说明

1. **原始版本** (`soft_unet_use_gui/`)
   - 分割网络：UNet
   - 预测网络：传统机器学习回归模型
   - 适用场景：使用传统机器学习方法进行重量预测

2. **ConvGRU版本** (`soft_unet_use_gui_convgru/`)
   - 分割网络：UNet
   - 预测网络：ConvGRU深度学习网络
   - 适用场景：使用深度学习ConvGRU网络进行重量预测

3. **VSSM版本** (`soft_unet_use_gui_vmamba/`)
   - 分割网络：VSSM (Visual State Space Model)
   - 预测网络：ConvGRU深度学习网络
   - 适用场景：使用VSSM进行分割，ConvGRU进行预测

#### 🚀 启动方式

```bash
# 原始版本（传统ML）
cd soft_unet_use_gui
python start.py

# ConvGRU版本
cd soft_unet_use_gui_convgru
python start.py

# VSSM版本
cd soft_unet_use_gui_vmamba
python start.py
# 或使用专用启动文件
python start_vmamba.py
```

#### 📋 详细使用步骤

##### 步骤1：启动程序
- 双击启动文件或在命令行中运行
- 程序会显示主界面，包含模型加载、路径设置、预测控制等区域

##### 步骤2：加载模型
1. **选择分割模型**
   - 点击第一个"open"按钮
   - 选择分割网络权重文件（.pth格式）
   - 支持UNet或VSSM模型权重

2. **选择预测模型**
   - 点击第二个"open"按钮
   - 选择预测模型权重文件
   - 支持格式：
     - `.pth`：ConvGRU深度学习模型
     - `.joblib`：传统机器学习模型

3. **加载模型**
   - 点击"Load model"按钮
   - 等待模型加载完成
   - 系统会自动识别模型类型

##### 步骤3：设置输入输出路径
1. **输入图片文件夹**
   - 点击"open"按钮选择包含待预测图片的文件夹
   - 支持PNG、JPG、JPEG格式
   - 文件夹结构：`根目录/样本文件夹/图片文件`

2. **输出结果文件夹**
   - 点击"open"按钮选择保存结果的文件夹
   - 程序会自动创建文件夹（如果不存在）

##### 步骤4：开始预测
1. **启动预测**
   - 点击"begin"按钮开始批量预测
   - 程序会显示"预测进行中，请稍候..."

2. **预测过程**
   - 系统自动处理每个样本
   - 显示处理进度和状态信息
   - 可以随时点击"Terminate"按钮停止

3. **查看结果**
   - 预测完成后，结果会显示在表格中
   - 包含样本名称、预测重量、排名等信息
   - 结果按重量从大到小排序

##### 步骤5：导出结果
1. **导出预测数据**
   - 点击"Export results"按钮
   - 选择保存文件夹
   - 结果自动保存为CSV格式

2. **输出文件命名规则**
   - 原始版本：`predict_result.csv`
   - ConvGRU版本：`predict_result_convgru.csv`
   - VSSM版本：`predict_result_vssm_convgru.csv`

#### 🔧 高级功能

##### 模型权重导出
ConvGRU和VSSM版本支持将训练好的模型导出为多种格式：

```python
# 在Python中使用
from models import PredictorConvGRU

predictor = PredictorConvGRU()
predictor.load_models("unet_model.pth", "convgru_model.pth")

# 导出为joblib格式（兼容传统ML接口）
predictor.save_model_as_joblib("convgru_model.joblib")

# 导出所有格式
predictor.export_model_weights("export_folder/")
```

##### 批量处理优化
- 支持多线程处理，避免界面卡顿
- 自动内存管理，处理大量图片
- 可中断和恢复预测过程

#### 📁 数据格式要求

##### 输入图片格式
- **文件格式**：PNG、JPG、JPEG
- **图片尺寸**：建议1536x1536或更大
- **颜色模式**：RGB或灰度
- **文件命名**：建议包含时间戳和尺寸信息

##### 文件夹结构
```
输入文件夹/
├── 样本1/
│   ├── DR/
│   │   ├── 2024.12.18_15.25.45_1536x1536.jpg
│   │   └── 2024.12.18_15.25.45_1536x1536.raw
│   └── ...
├── 样本2/
│   └── ...
└── ...
```

##### 输出结果格式
- **分割结果**：保存为PNG格式掩码图片
- **预测结果**：保存为CSV表格文件
- **日志信息**：实时显示在界面中

#### ⚠️ 注意事项

1. **模型兼容性**
   - 确保分割模型和预测模型匹配
   - 检查模型文件版本和结构
   - 验证输入数据预处理方式

2. **系统要求**
   - 内存：建议8GB以上
   - GPU：可选，用于加速推理
   - 存储：确保有足够空间保存结果

3. **错误处理**
   - 检查控制台输出的错误信息
   - 验证文件路径和权限
   - 确认依赖包版本兼容性

4. **性能优化**
   - 使用GPU加速（如果可用）
   - 调整批处理大小
   - 关闭不必要的后台程序

## 📁 项目结构

```
chicken-Mamba/
├── vmunet.py                    # 改进的VMUNet分割网络
├── gru.py                       # ConvGRU预测网络训练和评估
├── network1.py                  # ConvGRU + PAM网络结构定义
├── vmamba.py                    # VSSM核心模块
├── 机器学习提取特征文件.py        # 传统机器学习特征提取方法
├── README.txt                   # 项目主说明文档
├── GUI_VERSIONS_SUMMARY.md      # GUI版本功能总结
├── convert_models_to_joblib.py  # 模型格式转换工具
├── soft_unet_use_gui/          # 原始版本GUI（UNet + 传统ML）
│   ├── start.py                 # 主启动文件
│   ├── start.spec               # PyInstaller打包配置
│   ├── setup.py                 # 安装配置脚本
│   ├── 打包.txt                 # 打包说明
│   ├── models.py                # 传统ML预测器
│   ├── modules/
│   │   └── qt_thread.py         # 线程处理模块
│   ├── ui/                      # UI界面文件
│   └── README.md                # 原始版本使用说明
├── soft_unet_use_gui_convgru/  # ConvGRU版本GUI（UNet + ConvGRU）
│   ├── start.py                 # 主启动文件
│   ├── start.spec               # PyInstaller打包配置
│   ├── models.py                # ConvGRU预测器（支持joblib导出）
│   ├── modules/
│   │   └── qt_thread.py         # 线程处理模块
│   ├── ui/                      # UI界面文件
│   └── README_ConvGRU.md        # ConvGRU版本使用说明
└── soft_unet_use_gui_vmamba/   # VSSM版本GUI（VSSM + ConvGRU）
    ├── start.py                 # 主启动文件
    ├── start_vmamba.py          # VSSM专用启动文件
    ├── start.spec               # PyInstaller打包配置
    ├── models.py                # VSSM+ConvGRU预测器（支持joblib导出）
    ├── modules/
    │   └── qt_thread.py         # 线程处理模块
    ├── ui/                      # UI界面文件
    └── README_VSSM.md           # VSSM版本使用说明
```

## 🧠 网络架构

### VMUNet分割网络
- **基础架构**: 基于VSSM(Visual State Space Model)
- **增强模块**: 
  - MWT: 多尺度小波变换模块
  - DCF: 多尺度空洞卷积融合模块
- **特点**: 结合频域和空域特征，提升分割精度

### ConvGRU预测网络
- **架构**: ConvGRU + 位置注意力机制(PAM)
- **输入**: 图像序列
- **输出**: 连续数值预测
- **特点**: 处理时序信息，适合多帧预测

### 传统机器学习方法
- **特征提取**: 几何形状特征（面积、周长、椭圆率等）
- **算法**: 支持多种传统回归算法
- **用途**: 作为深度学习方法的对比基准

## 📦 软件打包

### 打包工具
本项目使用PyInstaller进行软件打包，可以将Python程序打包成独立的可执行文件（.exe），无需安装Python环境即可运行。

### 打包方法

#### 1. 安装PyInstaller
```bash
pip install pyinstaller
```

#### 2. 使用现有配置文件打包
每个GUI版本都包含预配置的打包文件：

**原始版本打包**：
```bash
cd soft_unet_use_gui
pyinstaller start.spec
```

**ConvGRU版本打包**：
```bash
cd soft_unet_use_gui_convgru
pyinstaller start.spec
```

**VSSM版本打包**：
```bash
cd soft_unet_use_gui_vmamba
pyinstaller start.spec
```

#### 3. 自定义打包配置
如果需要修改打包配置，可以编辑`.spec`文件：

```python
# 修改start.spec文件中的配置
a = Analysis(
    ['start.py'],                    # 主程序文件
    pathex=[],                       # 额外路径
    binaries=[],                     # 二进制文件
    datas=[],                        # 数据文件
    hiddenimports=[...],            # 隐藏导入
    excludes=[],                     # 排除模块
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChickenWeightPredictor',   # 可执行文件名
    debug=False,                      # 调试模式
    console=False,                    # 无控制台窗口
    icon=['ui\\icons\\bg.png'],      # 程序图标
)
```

#### 4. 打包后的文件结构
打包完成后，在`dist/`文件夹中会生成：
```
dist/
└── start/                          # 程序文件夹
    ├── start.exe                   # 主程序（Windows）
    ├── start                       # 主程序（Linux/Mac）
    ├── _internal/                  # 依赖库
    └── ui/                         # UI资源文件
```

#### 5. 分发软件
- **Windows**: 将整个`start`文件夹打包成zip文件
- **Linux/Mac**: 将整个`start`文件夹打包成tar.gz文件
- **用户使用**: 解压后直接运行可执行文件，无需安装Python

### 打包注意事项

1. **依赖管理**
   - 确保所有依赖包都已安装
   - 检查隐藏导入模块是否完整
   - 验证数据文件路径正确

2. **文件大小优化**
   - 使用`--exclude-module`排除不需要的模块
   - 压缩二进制文件（UPX）
   - 清理不必要的依赖

3. **跨平台兼容**
   - Windows: 生成.exe文件
   - Linux: 生成无扩展名文件
   - Mac: 生成.app包

4. **图标和资源**
   - 确保图标文件路径正确
   - 包含所有必要的UI资源
   - 验证数据文件完整性

### 快速打包命令

```bash
# 基础打包（自动检测依赖）
pyinstaller --onefile --windowed start.py

# 使用spec文件打包（推荐）
pyinstaller start.spec

# 打包成单文件
pyinstaller --onefile start.py

# 打包成文件夹（便于调试）
pyinstaller --onedir start.py
```

## 🔧 环境要求

### 核心依赖
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
PyQt5>=5.15.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.4.0
einops>=0.3.0
timm>=0.4.0
```

### 可选依赖
```
scipy>=1.7.0
tqdm>=4.62.0
mamba-ssm>=1.0.0
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认PyTorch版本兼容性
   - 验证模型文件完整性

2. **GUI界面无法启动**
   - 确认PyQt5安装正确
   - 检查系统图形界面支持
   - 验证Python环境配置

3. **预测结果异常**
   - 检查输入数据格式
   - 确认模型文件匹配
   - 验证数据预处理步骤

4. **内存不足**
   - 减小批处理大小
   - 使用CPU模式运行
   - 关闭其他占用内存的程序

### 错误日志
如果遇到问题，请检查控制台输出的错误信息，并确保：
- 所有依赖包已正确安装
- 模型文件路径正确
- 输入数据格式符合要求

## 🤝 贡献指南

我们欢迎社区贡献！如果您想为项目做出贡献，请：

1. Fork 本项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 贡献类型
- 🐛 Bug 修复
- ✨ 新功能
- 📚 文档改进
- 🎨 代码优化
- 🧪 测试用例

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目维护者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目链接: [https://github.com/your-username/chicken-Mamba](https://github.com/your-username/chicken-Mamba)

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究人员。

---

如果这个项目对您有帮助，请给我们一个 ⭐ Star！