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
- [Environment Requirements](#environment-requirements)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)  

## ✨ Features

- 🖼️ **Image Segmentation**: Improved VMUNet for segmentation  
- 🧠 **Weight Prediction**: ConvGRU for regression  
- 🔄 **Comparison**: Deep learning (ConvGRU) vs. traditional ML feature extraction  
- 📦 **Batch Processing**: Process multiple samples at once  
- 🖥️ **GUI**: PyQt5 graphical interface  
- 🔧 **Multi-Version**: Three independent GUI versions  
- 📤 **Model Export**: Export deep learning models to joblib  

## 🚀 Installation

### Requirements
- Python 3.7+  
- CUDA 11.0+ (optional, for GPU acceleration)  
- RAM: 8GB+  
- OS: Windows/Linux/macOS  

### Steps

1. **Clone the project**
```bash
git clone https://github.com/CAU-LEI/Chicken-Mamba.git
cd chicken-Mamba
