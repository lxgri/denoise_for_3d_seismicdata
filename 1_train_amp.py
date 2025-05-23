import os
import sys
import argparse
import segyio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ====================== 调试检查 ======================
def debug_checks(args):
    """检查环境、文件路径和CUDA可用性"""
    print("===== 环境检查 =====")
    print(f"Python路径: {sys.executable}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")

    print("\n===== 文件检查 =====")
    assert os.path.exists(args.noisy), f"含噪文件不存在: {args.noisy}"
    assert os.path.exists(args.clean), f"干净文件不存在: {args.clean}"
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    print("✅ 所有检查通过")

# ====================== 数据加载 ======================
class SeismicDataset(Dataset):
    def __init__(self, noisy_path, clean_path):
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.noisy_data = self._load_segy(noisy_path)
        self.clean_data = self._load_segy(clean_path)
        assert self.noisy_data.shape == self.clean_data.shape, "数据形状不匹配"

    def _load_segy(self, path):
        """加载SEGY文件并转换为numpy数组"""
        with segyio.open(path, ignore_geometry=True) as f:
            data = np.stack([trace.astype(np.float32) for trace in f.trace])
        return data

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        noisy = torch.from_numpy(self.noisy_data[idx])
        clean = torch.from_numpy(self.clean_data[idx])
        return noisy, clean

# ====================== 模型定义 ======================
class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, N)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

# ====================== 训练逻辑 ======================
def train(args):
    debug_checks(args)  # 调试检查

    # 数据加载
    dataset = SeismicDataset(args.noisy, args.clean)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"数据加载完成: {len(dataset)}个样本")

    # 模型初始化
    model = DenoisingModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for noisy, clean in dataloader:
            noisy, clean = noisy.cuda(), clean.cuda()
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

    # 保存模型
    torch.save(model.state_dict(), args.output_model)
    print(f"模型已保存到: {args.output_model}")

# ====================== 主程序 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", type=str, required=True, help="含噪SEGY文件路径")
    parser.add_argument("--clean", type=str, required=True, help="干净SEGY文件路径")
    parser.add_argument("--output_model", type=str, required=True, help="输出模型路径")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}", file=sys.stderr)
        sys.exit(1)