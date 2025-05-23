#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import segyio
import os
import tempfile
import shutil
from tqdm import tqdm

# ====== 模型定义（必须与训练一致） ======
class DenoisingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ====== 数据加载 ======
def load_segy(path, primary=9, secondary=13):
    """按 (字节9, 字节13) 排序加载数据"""
    with segyio.open(path, ignore_geometry=True) as f:
        indices = sorted(
            range(len(f.trace)),
            key=lambda i: (f.header[i][primary], f.header[i][secondary])  # 修改点
        )
        data = np.stack([f.trace[i].astype(np.float32) for i in indices])
        return data, indices

# ====== 原子化写入 ======
def atomic_write_segy(data, input_path, output_path, orig_indices, mean, std):
    """按原始顺序保存并还原标准化"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, os.path.basename(output_path))
    
    try:
        with segyio.open(input_path, ignore_geometry=True) as src:
            spec = segyio.spec()
            spec.format = src.format
            spec.samples = src.samples
            spec.tracecount = len(data)
            
            with segyio.create(temp_path, spec) as dst:
                # 还原标准化并保存
                for new_pos, old_pos in enumerate(orig_indices):
                    dst.trace[new_pos] = data[old_pos] * std + mean  # 反标准化
                    dst.header[new_pos] = src.header[old_pos]
                
                dst.text[0] = src.text[0]
                dst.bin = src.bin
        
        if os.path.exists(output_path):
            os.unlink(output_path)
        shutil.move(temp_path, output_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ====== 主函数 ======
def inference(input_model, input_noisy, output_denoised, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和配置
    checkpoint = torch.load(input_model, map_location=device)
    config = checkpoint['config']
    model = DenoisingModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据（按训练相同规则排序）
    noisy_data, orig_indices = load_segy(
        input_noisy,
        primary=config['primary_header'],
        secondary=config['secondary_header']
    )
    
    # 标准化（与训练一致）
    noisy_data