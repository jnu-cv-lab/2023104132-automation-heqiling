import math
import torch
import numpy as np
import matplotlib
# 适配WSL无图形界面，自动保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 全局超参数配置 =====================
d_model = 64    # 词嵌入维度（高维向量）
max_len = 20    # 最大序列长度
batch_size = 1  # 批次大小

# ===================== 任务1：实现 Sinusoidal 正弦位置编码 =====================
def sinusoidal_pe(max_seq_len: int, embed_dim: int) -> torch.Tensor:
    """
    标准Transformer正弦位置编码
    :param max_seq_len: 最大序列长度
    :param embed_dim: 词嵌入维度
    :return: 位置编码矩阵 (max_seq_len, embed_dim)
    """
    pe = torch.zeros(max_seq_len, embed_dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    # 计算频率项 10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    
    # 奇偶维度分别使用 sin / cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# 生成并可视化正弦位置编码
pe_sin = sinusoidal_pe(max_len, d_model)
print("=" * 60)
print("【任务1 正弦位置编码 Sinusoidal PE】")
print(f"位置编码矩阵形状: {pe_sin.shape}")
print(f"位置0 前8维编码值: {pe_sin[0, :8].numpy()}")
print(f"位置5 前8维编码值: {pe_sin[5, :8].numpy()}")

# 绘制正弦编码曲线
plt.figure(figsize=(10, 5))
for i in range(0, 8, 2):
    plt.plot(range(max_len), pe_sin[:, i].numpy(), label=f"dim {i}(sin)")
    plt.plot(range(max_len), pe_sin[:, i+1].numpy(), label=f"dim {i+1}(cos)")
plt.title("Sinusoidal Position Encoding Curve")
plt.xlabel("Position")
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.savefig("sin_pe_curve.png", dpi=150)
plt.close()

# ===================== 任务2：实现 二维向量旋转 =====================
def rotate_2d(vec: torch.Tensor, theta: float) -> torch.Tensor:
    """
    标准二维向量旋转
    :param vec: 二维向量 (2,)
    :param theta: 旋转角度（弧度）
    :return: 旋转后的二维向量
    """
    # 二维旋转矩阵 [cosθ, -sinθ; sinθ, cosθ]
    rot_mat = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])
    return torch.matmul(rot_mat, vec)

# 测试二维旋转
print("\n" + "=" * 60)
print("【任务2 二维向量旋转测试】")
test_vec_2d = torch.tensor([1.0, 0.0])  # 初始向量 (1,0)
theta = math.pi / 4  # 旋转45度
rot_vec_2d = rotate_2d(test_vec_2d, theta)
print(f"原始二维向量: {test_vec_2d.numpy()}")
print(f"旋转45度后向量: {np.round(rot_vec_2d.numpy(), 4)}")

# ===================== 任务3：实现 高维 RoPE 旋转位置编码 =====================
def rope_encoding(x: torch.Tensor, pos: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    高维RoPE旋转位置编码（逐二维分组旋转）
    :param x: 输入向量 (batch, seq_len, embed_dim)
    :param pos: 位置索引 (seq_len,)
    :param embed_dim: 嵌入维度（必须为偶数）
    :return: 加入RoPE后的向量
    """
    assert embed_dim % 2 == 0, "RoPE要求嵌入维度为偶数"
    seq_len = x.shape[1]
    theta_base = 10000.0

    # 初始化RoPE输出
    x_rope = x.clone()
    for i in range(0, embed_dim, 2):
        # 每组二维对应的频率（与正弦PE频率一致）
        freq = 1.0 / (theta_base ** (i / embed_dim))
        for t in range(seq_len):
            pos_t = pos[t]
            theta_t = pos_t * freq
            # 取出当前二维向量
            x1 = x[:, t, i]
            x2 = x[:, t, i+1]
            # 二维旋转公式
            x_rope[:, t, i] = x1 * math.cos(theta_t) - x2 * math.sin(theta_t)
            x_rope[:, t, i+1] = x1 * math.sin(theta_t) + x2 * math.cos(theta_t)
    return x_rope

# 测试高维RoPE
print("\n" + "=" * 60)
print("【任务3 高维RoPE测试】")
# 模拟词嵌入向量 (batch=1, seq_len=max_len, dim=d_model)
token_emb = torch.randn(batch_size, max_len, d_model)
pos_idx = torch.arange(0, max_len)  # 位置索引 0~19
emb_rope = rope_encoding(token_emb, pos_idx, d_model)
print(f"原始词嵌入形状: {token_emb.shape}")
print(f"RoPE编码后形状: {emb_rope.shape}")
print(f"位置3 前8维RoPE值: {emb_rope[0, 3, :8].numpy().round(4)}")

# ===================== 任务4：对比 E+pos 与 RoPE 输入方式 =====================
print("\n" + "=" * 60)
print("【任务4 E+pos 与 RoPE 输入方式对比】")
# 方式1：传统 E+pos（嵌入 + 正弦位置编码，逐元素相加）
emb_plus_pe = token_emb + pe_sin[:max_len, :]
print("1. 传统 E+pos 输入方式: 词嵌入 + 正弦编码 (逐元素加法)")
print(f"E+pos 输出形状: {emb_plus_pe.shape}")

# 方式2：RoPE 输入方式（对Q/K向量做旋转变换，无加法）
print("2. RoPE 输入方式: 对Query/Key向量执行二维分组旋转 (无加法)")
print("核心差异: E+pos是「加法融合」，RoPE是「几何旋转融合」")

# ===================== 任务5：数值实验 验证RoPE相对位置性质 =====================
print("\n" + "=" * 60)
print("【任务5 数值实验：验证RoPE相对位置特性】")
# 实验设计：Q、K向量固定，改变绝对位置，保持相对位置不变
# 定义基础Q/K
q_base = torch.randn(1, 1, d_model)
k_base = torch.randn(1, 1, d_model)

# 实验组1：相对距离=3，绝对位置 (pos_q=2, pos_k=5)
pos_q1, pos_k1 = 2, 5
q1 = rope_encoding(q_base, torch.tensor([pos_q1]), d_model)
k1 = rope_encoding(k_base, torch.tensor([pos_k1]), d_model)
score1 = torch.matmul(q1, k1.transpose(-2, -1)).item()

# 实验组2：相对距离=3，绝对位置整体偏移 (pos_q=4, pos_k=7)
pos_q2, pos_k2 = 4, 5
pos_q2, pos_k2 = 4, 7
q2 = rope_encoding(q_base, torch.tensor([pos_q2]), d_model)
k2 = rope_encoding(k_base, torch.tensor([pos_k2]), d_model)
score2 = torch.matmul(q2, k2.transpose(-2, -1)).item()

# 实验组3：绝对位置不变，相对距离改为4 (pos_q=2, pos_k=6)
pos_q3, pos_k3 = 2, 6
q3 = rope_encoding(q_base, torch.tensor([pos_q3]), d_model)
k3 = rope_encoding(k_base, torch.tensor([pos_k3]), d_model)
score3 = torch.matmul(q3, k3.transpose(-2, -1)).item()

# 打印数值结果
print(f"实验组1(相对距离3, 位置2&5) 注意力分数: {score1:.4f}")
print(f"实验组2(相对距离3, 位置4&7) 注意力分数: {score2:.4f}")
print(f"实验组3(相对距离4, 位置2&6) 注意力分数: {score3:.4f}")
print("结论: 相对位置相同 → 分数几乎一致；相对位置改变 → 分数明显变化")

# 可视化验证结果
plt.figure(figsize=(8, 4))
x = ["相对距离3(组1)", "相对距离3(组2)", "相对距离4(组3)"]
y = [score1, score2, score3]
plt.bar(x, y, color=["#1f77b4", "#1f77b4", "#ff7f0e"])
plt.title("RoPE 相对位置验证")
plt.ylabel("Q-K 点积分数(Attention Score)")
plt.savefig("rope_verify.png", dpi=150)
plt.close()

# ===================== 实验结束 =====================
print("\n" + "=" * 60)
print("✅ 所有任务执行完成！图片已保存至当前目录")