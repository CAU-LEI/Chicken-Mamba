#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型转换工具：将训练好的深度学习模型转换为joblib格式
支持ConvGRU和VSSM+ConvGRU模型
"""

import os
import sys
import torch
import joblib
import torch.nn.functional as F
from pathlib import Path

# 添加项目路径
sys.path.append('.')

def convert_convgru_to_joblib(convgru_model_path, output_dir):
    """
    将训练好的ConvGRU模型转换为joblib格式
    
    Args:
        convgru_model_path: ConvGRU模型文件路径(.pth)
        output_dir: 输出目录
    """
    try:
        from network1 import ConvGRURegression
        
        # 创建ConvGRU模型实例
        convgru = ConvGRURegression(
            input_size=(64, 64),
            input_dim=3,
            hidden_dim=[64, 128],
            kernel_size=(3, 3),
            num_layers=2,
            output_dim=1
        )
        
        # 加载训练好的权重
        convgru.load_state_dict(torch.load(convgru_model_path, map_location='cpu'))
        convgru.eval()
        
        # 创建包装器类
        class ConvGRUWrapper:
            def __init__(self, convgru_model):
                self.model = convgru_model
                self.input_size = (64, 64)
                self.input_dim = 3
                self.hidden_dim = [64, 128]
                self.kernel_size = (3, 3)
                self.num_layers = 2
                self.output_dim = 1
                self.model_type = "ConvGRU"
            
            def predict(self, features):
                """
                预测方法，兼容joblib模型接口
                """
                with torch.no_grad():
                    # 转换特征为ConvGRU期望的格式
                    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    feature_tensor = feature_tensor.unsqueeze(0)
                    feature_tensor = F.interpolate(feature_tensor, size=(64, 64), mode='bilinear', align_corners=False)
                    feature_tensor = feature_tensor.repeat(1, 5, 1, 1, 1)
                    weight_pred = self.model(feature_tensor)
                    return [weight_pred.item()]
            
            def get_info(self):
                """
                获取模型信息
                """
                return {
                    "model_type": self.model_type,
                    "input_size": self.input_size,
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "kernel_size": self.kernel_size,
                    "num_layers": self.num_layers,
                    "output_dim": self.output_dim
                }
        
        # 创建包装器并保存
        wrapper = ConvGRUWrapper(convgru)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为joblib格式
        output_path = os.path.join(output_dir, "convgru_model.joblib")
        joblib.dump(wrapper, output_path)
        
        print(f"✅ ConvGRU模型成功转换为joblib格式")
        print(f"   输出路径: {output_path}")
        print(f"   模型信息: {wrapper.get_info()}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ ConvGRU模型转换失败: {e}")
        return None

def convert_vssm_convgru_to_joblib(vssm_model_path, convgru_model_path, output_dir):
    """
    将训练好的VSSM+ConvGRU模型转换为joblib格式
    
    Args:
        vssm_model_path: VSSM模型文件路径(.pth)
        convgru_model_path: ConvGRU模型文件路径(.pth)
        output_dir: 输出目录
    """
    try:
        from network1 import ConvGRURegression
        from vmamba import VSSM
        
        # 创建VSSM模型实例
        vssm = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1,
            depths=[2, 2, 9, 2],
            depths_decoder=[2, 9, 2, 2],
            dims=[96, 192, 384, 768],
            dims_decoder=[768, 384, 192, 96],
            d_state=16,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_checkpoint=False
        )
        
        # 创建ConvGRU模型实例
        convgru = ConvGRURegression(
            input_size=(64, 64),
            input_dim=3,
            hidden_dim=[64, 128],
            kernel_size=(3, 3),
            num_layers=2,
            output_dim=1
        )
        
        # 加载训练好的权重
        vssm.load_state_dict(torch.load(vssm_model_path, map_location='cpu'))
        convgru.load_state_dict(torch.load(convgru_model_path, map_location='cpu'))
        
        vssm.eval()
        convgru.eval()
        
        # 创建包装器类
        class VSSMConvGRUWrapper:
            def __init__(self, vssm_model, convgru_model):
                self.vssm = vssm_model
                self.convgru = convgru_model
                self.input_size = (64, 64)
                self.input_dim = 3
                self.hidden_dim = [64, 128]
                self.kernel_size = (3, 3)
                self.num_layers = 2
                self.output_dim = 1
                self.model_type = "VSSM+ConvGRU"
            
            def predict(self, features):
                """
                预测方法，兼容joblib模型接口
                """
                with torch.no_grad():
                    # 转换特征为ConvGRU期望的格式
                    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    feature_tensor = feature_tensor.unsqueeze(0)
                    feature_tensor = F.interpolate(feature_tensor, size=(64, 64), mode='bilinear', align_corners=False)
                    feature_tensor = feature_tensor.repeat(1, 5, 1, 1, 1)
                    weight_pred = self.convgru(feature_tensor)
                    return [weight_pred.item()]
            
            def get_info(self):
                """
                获取模型信息
                """
                return {
                    "model_type": self.model_type,
                    "input_size": self.input_size,
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "kernel_size": self.kernel_size,
                    "num_layers": self.num_layers,
                    "output_dim": self.output_dim
                }
        
        # 创建包装器并保存
        wrapper = VSSMConvGRUWrapper(vssm, convgru)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为joblib格式
        output_path = os.path.join(output_dir, "vssm_convgru_model.joblib")
        joblib.dump(wrapper, output_path)
        
        print(f"✅ VSSM+ConvGRU模型成功转换为joblib格式")
        print(f"   输出路径: {output_path}")
        print(f"   模型信息: {wrapper.get_info()}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ VSSM+ConvGRU模型转换失败: {e}")
        return None

def main():
    """
    主函数
    """
    print("🔧 深度学习模型转Joblib格式工具")
    print("=" * 50)
    
    while True:
        print("\n请选择转换类型:")
        print("1. ConvGRU模型 → Joblib格式")
        print("2. VSSM+ConvGRU模型 → Joblib格式")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            # ConvGRU模型转换
            convgru_path = input("请输入ConvGRU模型文件路径(.pth): ").strip()
            if not os.path.exists(convgru_path):
                print("❌ 文件不存在，请检查路径")
                continue
                
            output_dir = input("请输入输出目录: ").strip()
            if not output_dir:
                output_dir = "converted_models"
            
            convert_convgru_to_joblib(convgru_path, output_dir)
            
        elif choice == "2":
            # VSSM+ConvGRU模型转换
            vssm_path = input("请输入VSSM模型文件路径(.pth): ").strip()
            if not os.path.exists(vssm_path):
                print("❌ VSSM文件不存在，请检查路径")
                continue
                
            convgru_path = input("请输入ConvGRU模型文件路径(.pth): ").strip()
            if not os.path.exists(convgru_path):
                print("❌ ConvGRU文件不存在，请检查路径")
                continue
                
            output_dir = input("请输入输出目录: ").strip()
            if not output_dir:
                output_dir = "converted_models"
            
            convert_vssm_convgru_to_joblib(vssm_path, convgru_path, output_dir)
            
        elif choice == "3":
            print("👋 再见！")
            break
            
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()
