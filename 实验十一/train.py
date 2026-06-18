import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # WSL无GUI适配
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

# ===================== 超参数配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 132
T_FRAME = 30
D_MODEL = 128
N_HEAD = 4
N_LAYER = 2
FF_DIM = 256
NUM_CLS = 6
DROPOUT = 0.1
BATCH_SIZE = 16
LR = 1e-3
EPOCH = 20
DATA_DIR = "skeleton_npy"
MODEL_SAVE_PATH = "model_output/badminton_transformer.pth"
PIC_SAVE_DIR = "pic_output"

# ===================== 数据集类 =====================
class SkeletonDataset(Dataset):
    def __init__(self, x_np, y_np):
        self.x = torch.from_numpy(x_np).float()
        self.y = torch.from_numpy(y_np).long()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ===================== 正弦位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        return x + self.pe[:x.shape[1], :].unsqueeze(0)

# ===================== Skeleton Transformer 模型 =====================
class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入嵌入
        self.embedding = nn.Linear(INPUT_DIM, D_MODEL)
        # 位置编码
        self.pos_enc = PositionalEncoding(T_FRAME, D_MODEL)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYER)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, NUM_CLS)
        )

    def forward(self, x):
        # x: [B, 30, 132]
        x = self.embedding(x)       # [B, 30, 128]
        x = self.pos_enc(x)         # 加入位置信息
        x = self.encoder(x)         # Transformer编码
        pool_feat = torch.mean(x, dim=1)  # 全局均值池化
        out = self.classifier(pool_feat)  # 分类输出
        return out

# ===================== 主训练流程 =====================
def main():
    # 创建输出目录
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(PIC_SAVE_DIR, exist_ok=True)

    # 加载数据
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_map.json"), "r", encoding="utf-8") as f:label_map = json.load(f)
    # 动态获取数据中实际存在的类别，自动适配类别数量
    unique_labels = sorted(np.unique(y_train).tolist())
    class_names = [label_map[str(lb)] for lb in unique_labels]
    print(f"实际数据包含 {len(unique_labels)} 个类别：{class_names}")

    # 构建数据加载器
    train_set = SkeletonDataset(X_train, y_train)
    test_set = SkeletonDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 模型、损失、优化器
    model = SkeletonTransformer().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    best_acc = 0.0

    print(f"使用设备：{DEVICE}")
    print("开始训练...\n")

    # 训练循环
    for epoch in range(EPOCH):
        # ===== 训练阶段 =====
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y).sum().item()
            total += len(y)
        
        tr_loss = total_loss / len(train_loader)
        tr_acc = correct / total
        train_loss_list.append(tr_loss)
        train_acc_list.append(tr_acc)

        # ===== 测试阶段 =====
        model.eval()
        total_tloss, t_corr, t_total = 0.0, 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = loss_fn(logits, y)
                total_tloss += loss.item()
                pred = torch.argmax(logits, dim=-1)
                t_corr += (pred == y).sum().item()
                t_total += len(y)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        
        te_loss = total_tloss / len(test_loader)
        te_acc = t_corr / t_total
        test_loss_list.append(te_loss)
        test_acc_list.append(te_acc)

        print(f"Epoch {epoch+1:02d}/{EPOCH} | "
              f"Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | "
              f"Test Loss:{te_loss:.4f} Acc:{te_acc:.4f}")

        # 保存最优模型
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n训练结束！最优测试准确率：{best_acc:.4f}")

    # ===== 绘制训练曲线（已修复subplot参数） =====
    plt.figure(figsize=(12, 5))
    # 左图：Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss", linewidth=2)
    plt.plot(test_loss_list, label="Test Loss", linewidth=2)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # 右图：Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc", linewidth=2)
    plt.plot(test_acc_list, label="Test Acc", linewidth=2)
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_SAVE_DIR, "train_curve.png"), dpi=150)
    plt.close()

    # ===== 绘制混淆矩阵 =====
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_SAVE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ===== 打印分类报告 =====
    print("\n" + "="*50)
    print("测试集分类报告：")
    print(classification_report(all_true, all_pred, labels=unique_labels, target_names=class_names))

if __name__ == "__main__":
    main()