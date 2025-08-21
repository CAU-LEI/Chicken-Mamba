# GUI版本功能总结

## 概述
现在项目包含三个独立的GUI版本，每个版本都支持不同的模型组合，并且都支持机器学习模型权重的加载。

## 版本对比

### 1. 原始版本 (`soft_unet_use_gui/`)
- **分割网络**: UNet
- **预测网络**: 传统机器学习回归模型
- **模型文件**: 
  - 分割模型: `.pth` (UNet权重)
  - 预测模型: `.joblib` (机器学习模型权重)
- **输出文件**: `predict_result.csv`
- **特点**: 使用传统机器学习方法进行重量预测

### 2. ConvGRU版本 (`soft_unet_use_gui_convgru/`)
- **分割网络**: UNet
- **预测网络**: ConvGRU 或 传统机器学习回归模型
- **模型文件**: 
  - 分割模型: `.pth` (UNet权重)
  - 预测模型: `.pth` (ConvGRU权重) 或 `.joblib` (机器学习模型权重)
- **输出文件**: 
  - ConvGRU: `predict_result_convgru.csv`
  - ML: `predict_result_ml.csv`
- **特点**: 自动检测模型类型，支持两种预测方法

### 3. VSSM版本 (`soft_unet_use_gui_vmamba/`)
- **分割网络**: VSSM
- **预测网络**: ConvGRU 或 传统机器学习回归模型
- **模型文件**: 
  - 分割模型: `.pth` (VSSM权重)
  - 预测模型: `.pth` (ConvGRU权重) 或 `.joblib` (机器学习模型权重)
- **输出文件**: 
  - ConvGRU: `predict_result_vssm_convgru.csv`
  - ML: `predict_result_vssm_ml.csv`
- **特点**: 使用VSSM进行分割，自动检测预测模型类型

## 机器学习模型权重支持

### 所有版本都支持
- **分割模型权重**: `.pth` 格式
- **机器学习模型权重**: `.joblib` 格式

### 自动检测机制
- 根据文件扩展名自动识别模型类型
- `.pth` → ConvGRU网络
- `.joblib` → 传统机器学习模型
- 无需手动选择，系统自动适配

## 使用方法

### 启动方式
```bash
# 原始版本
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

### 模型加载步骤
1. **选择分割模型**: 点击第一个"open"按钮，选择分割网络权重文件
2. **选择预测模型**: 点击第二个"open"按钮，选择预测模型权重文件
   - 支持 `.pth` (ConvGRU) 或 `.joblib` (ML) 格式
3. **加载模型**: 点击"Load model"按钮
4. **系统自动识别**: 根据文件扩展名自动选择预测方法

### 输出文件命名规则
- **原始版本**: `predict_result.csv`
- **ConvGRU版本**: 
  - ConvGRU: `predict_result_convgru.csv`
  - ML: `predict_result_ml.csv`
- **VSSM版本**: 
  - ConvGRU: `predict_result_vssm_convgru.csv`
  - ML: `predict_result_vssm_ml.csv`

## 技术特点

### 智能模型检测
- 无需用户手动指定模型类型
- 根据文件扩展名自动识别
- 支持混合使用不同模型

### 统一接口
- 三个版本使用相同的UI布局
- 一致的按钮和控件命名
- 相同的操作流程

### 灵活配置
- 可以自由组合分割网络和预测网络
- 支持传统方法和深度学习方法
- 便于模型对比和实验

## 优势

1. **灵活性**: 支持多种模型组合
2. **易用性**: 自动检测模型类型，无需手动配置
3. **兼容性**: 支持现有的机器学习模型权重
4. **扩展性**: 易于添加新的模型类型
5. **对比性**: 便于比较不同方法的性能

## 注意事项

1. 确保模型文件格式正确
2. 机器学习模型权重必须是 `.joblib` 格式
3. 深度学习模型权重必须是 `.pth` 格式
4. 分割网络和预测网络需要匹配
5. 建议使用相同的数据预处理流程

## 未来扩展

- 支持更多模型格式
- 添加模型性能对比功能
- 支持模型集成和投票
- 添加模型可视化功能
- 支持在线模型更新
