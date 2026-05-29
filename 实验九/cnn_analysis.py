import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===================== 设备选择 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备：", device)

# ===================== 数据加载与划分 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 划分训练/验证集 8:2
train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===================== 复用CNN模型（与上次实验一致） =====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 钩子函数获取特征图
feature_map = None
def hook_fn(module, input, output):
    global feature_map
    feature_map = output.detach().cpu()

# ===================== 通用训练函数 =====================
def train_model(optimizer_name, lr, epochs=5):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "SGD_Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        train_loss /= len(train_loader)
        train_acc = correct / total

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                out = model(img)
                loss = criterion(out, label)
                val_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / total

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    # 测试集评估
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    error_imgs = []
    error_true = []
    error_pred = []
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            test_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            all_pred.extend(pred.cpu().numpy())
            all_label.extend(label.cpu().numpy())
            # 收集错误样本
            mask = pred != label
            if mask.any():
                error_imgs.extend(img[mask].cpu())
                error_true.extend(label[mask].cpu().numpy())
                error_pred.extend(pred[mask].cpu().numpy())
    test_loss /= len(test_loader)
    test_acc = correct / total

    return model, train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_acc, all_pred, all_label, error_imgs, error_true, error_pred

# ===================== 任务2：三种优化器对比 =====================
print("========== 任务2：优化器对比 ==========")
opt_list = ["SGD", "SGD_Momentum", "Adam"]
opt_result = {}
for opt in opt_list:
    _, tl, ta, vl, va, tacc, _, _, _, _, _ = train_model(opt, lr=0.001)
    opt_result[opt] = {"train_loss":tl, "train_acc":ta, "val_loss":vl, "val_acc":va, "test_acc":tacc}
    print(f"{opt} 测试准确率：{tacc:.4f}")

# 绘制优化器对比曲线（修复subplot参数）
plt.figure(figsize=(12,5))
# 第1个图：Loss对比
plt.subplot(1, 2, 1)  # 1行2列，第1个图
for name, res in opt_result.items():
    plt.plot(res["train_loss"], label=name)
plt.title("优化器 Train Loss 对比")
plt.legend()
# 第2个图：Acc对比
plt.subplot(1, 2, 2)  # 1行2列，第2个图
for name, res in opt_result.items():
    plt.plot(res["train_acc"], label=name)
plt.title("优化器 Train Acc 对比")
plt.legend()
plt.savefig("optimizer_compare.png", dpi=150)
plt.close()

# ===================== 任务3：学习率对比（Adam） =====================
print("\n========== 任务3：学习率对比 ==========")
lr_list = [0.1, 0.01, 0.001]
lr_result = {}
for lr in lr_list:
    _, tl, ta, vl, va, tacc, _, _, _, _, _ = train_model("Adam", lr=lr)
    lr_result[lr] = {"train_loss":tl, "train_acc":ta, "val_loss":vl, "val_acc":va, "test_acc":tacc}
    print(f"学习率 {lr} 测试准确率：{tacc:.4f}")

# 绘制学习率对比曲线（已修复subplot参数）
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
for lr, res in lr_result.items():
    plt.plot(res["train_loss"], label=f"lr={lr}")
plt.title("学习率 Train Loss 对比")
plt.legend()

plt.subplot(1, 2, 2)
for lr, res in lr_result.items():
    plt.plot(res["train_acc"], label=f"lr={lr}")
plt.title("学习率 Train Acc 对比")
plt.legend()
plt.savefig("lr_compare.png", dpi=150)
plt.close()

# -------------------------- 关键修复：保存最优模型 --------------------------
# 用Adam + lr=0.001训练一个模型，保存model变量用于后续可视化
print("\n========== 保存最优模型 ==========")
model, _, _, _, _, _, all_pred, all_label, error_imgs, error_true, error_pred = train_model("Adam", lr=0.001)

# ===================== 任务4：卷积核可视化 =====================
print("\n========== 任务4：卷积核可视化 ==========")
conv1_weights = model.conv1.weight.detach().cpu().numpy()
plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(conv1_weights[i,0], cmap='gray')
    plt.title(f"Kernel {i+1}")
    plt.axis('off')
plt.savefig("conv_kernel.png", dpi=150)
plt.close()

# ===================== 任务5：特征图可视化 =====================
print("\n========== 任务5：特征图可视化 ==========")
# 钩子函数获取特征图
feature_map = None
def hook_fn(module, input, output):
    global feature_map
    feature_map = output.detach().cpu()

# 注册钩子
handle = model.conv1.register_forward_hook(hook_fn)
# 取一张测试图片
img_sample, _ = next(iter(test_loader))
img_sample = img_sample[0:1].to(device)
# 前向传播获取特征图
with torch.no_grad():
    model(img_sample)
handle.remove()

# 可视化8张特征图
plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(feature_map[0,i], cmap='gray')
    plt.title(f"FM {i+1}")
    plt.axis('off')
plt.savefig("feature_map.png", dpi=150)
plt.close()

# ===================== 任务6：错误样本可视化 =====================
print("\n========== 任务6：错误样本可视化 ==========")
plt.figure(figsize=(10,5))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(error_imgs[i].squeeze(), cmap='gray')
    plt.title(f"T:{error_true[i]}\nP:{error_pred[i]}")
    plt.axis('off')
plt.savefig("error_samples.png", dpi=150)
plt.close()

# ===================== 任务7：混淆矩阵 =====================
print("\n========== 任务7：混淆矩阵 ==========")
cm = confusion_matrix(all_label, all_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("MNIST 混淆矩阵")
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\n✅ 所有任务执行完成，图片已全部保存！")