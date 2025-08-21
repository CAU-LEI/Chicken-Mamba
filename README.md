# Chicken-Mamba
Chicken-Mamba: AI Phenotyping Pipeline for Chicken Testis Analysis


# Chicken-Mamba

A deep learning-based image segmentation and weight prediction system, combining an improved **VMUNet** segmentation network and **ConvGRU** prediction network, with traditional machine learning methods provided for comparison.

## 📋 Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Network Architecture](#network-architecture)  
- [Packaging](#packaging)  
- [Environment Requirements](#environment-requirements)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## ✨ Features

- 🖼️ **Image Segmentation**: Improved VMUNet  
- 🧠 **Weight Prediction**: ConvGRU regression  
- 🔄 **Comparison**: Deep learning (ConvGRU) vs. traditional ML  
- 📦 **Batch Processing**: Multi-sample support  
- 🖥️ **GUI**: PyQt5 graphical interface  
- 🔧 **Multi-Version**: Three GUI versions  
- 📤 **Model Export**: Export DL models as joblib  

## 🚀 Installation

### Requirements
- Python 3.7+  
- CUDA 11.0+ (optional, GPU acceleration)  
- RAM: 8GB+  
- OS: Windows/Linux/macOS  

### Steps

1. **Clone the project**
```bash
git clone https://github.com/CAU-LEI/Chicken-Mamba.git
cd chicken-Mamba
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Segmentation
```bash
python vmunet.py
```

### 2. Weight Prediction
```bash
python gru.py
```

### 3. GUI Versions
**Original (UNet + ML)**  
```bash
cd soft_unet_use_gui
python start.py
```

**ConvGRU (UNet + ConvGRU)**  
```bash
cd soft_unet_use_gui_convgru
python start.py
```

**VSSM+ConvGRU (VMUNet + ConvGRU)**  
```bash
cd soft_unet_use_gui_vmamba
python start.py
# or
python start_vmamba.py
```

### GUI Workflow
1. Start program → choose version  
2. Load models → segmentation (.pth) + prediction (.pth/.joblib)  
3. Set paths → input + output folders  
4. Start prediction → click **begin**  
5. Export results → CSV  

### Model Compatibility

| GUI Version | Segmentation | Prediction | Output |
|-------------|--------------|------------|--------|
| Original    | UNet (.pth)  | ML (.joblib) | CSV |
| ConvGRU     | UNet (.pth)  | ConvGRU (.pth) | CSV |
| VSSM        | VSSM (.pth)  | ConvGRU (.pth) | CSV |

All versions support exporting DL models to joblib.

## 📖 Usage

### Command Line
```bash
python vmunet.py   # segmentation
python gru.py      # prediction
```

### GUI Options
- **Original**: UNet + ML  
- **ConvGRU**: UNet + ConvGRU  
- **VSSM**: VSSM + ConvGRU  

## 📁 Project Structure

```
chicken-Mamba/
├── vmunet.py                    # VMUNet segmentation
├── gru.py                       # ConvGRU predictor
├── network1.py                  # ConvGRU + PAM
├── vmamba.py                    # VSSM core
├── ml_feature_extraction.py     # ML feature extraction
├── convert_models_to_joblib.py  # Format converter
├── soft_unet_use_gui/           # Original GUI
├── soft_unet_use_gui_convgru/   # ConvGRU GUI
└── soft_unet_use_gui_vmamba/    # VSSM GUI
```

## 🧠 Network Architecture

### VMUNet
- Based on **VSSM**  
- MWT (multi-scale wavelet)  
- DCF (dilated convolution fusion)  
- Combines frequency + spatial features  

### ConvGRU
- ConvGRU + PAM (position attention)  
- Input: image sequence  
- Output: regression weight  

### Traditional ML
- Features: area, perimeter, eccentricity  
- Regression algorithms as baseline  

## 📦 Packaging

Use **PyInstaller**:

```bash
pip install pyinstaller
pyinstaller start.spec
```

### Output
```
dist/
└── start/
    ├── start.exe   # Windows
    ├── start       # Linux/Mac
    ├── _internal/
    └── ui/
```

### Quick Commands
```bash
pyinstaller --onefile --windowed start.py
pyinstaller start.spec
```

## 🔧 Environment Requirements

### requirements.txt
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

# Optional
scipy>=1.7.0
tqdm>=4.62.0
mamba-ssm>=1.0.0
```

## 🐛 Troubleshooting

- **Model load fail**: check path, PyTorch version, file integrity  
- **GUI fail**: check PyQt5, OS GUI support  
- **Abnormal results**: check input format, preprocessing  
- **Memory issues**: reduce batch size, use CPU  

## 🤝 Contributing

1. Fork the repo  
2. Create branch (`git checkout -b feature/Name`)  
3. Commit (`git commit -m "Add feature"`)  
4. Push (`git push origin feature/Name`)  
5. Open PR  

Contribution types:  
- 🐛 Bug fixes  
- ✨ Features  
- 📚 Docs  
- 🎨 Optimization  
- 🧪 Tests  

## 📄 License

MIT License – see [LICENSE](LICENSE)  

## 📞 Contact

- Maintainer: **Lei Wei**  
- Email: **leiwei@cau.edu.cn**  
- Repo: [https://github.com/CAU-LEI/Chicken-Mamba.git](https://github.com/CAU-LEI/Chicken-Mamba.git)  
