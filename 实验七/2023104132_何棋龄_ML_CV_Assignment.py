# ==============================================
# 计算机视觉第8课：传统机器学习图像分类
# 数据集：sklearn.datasets.digits（8×8手写数字）
# 无弹窗版 · 自动保存图片
# ==============================================
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 关键：不弹出窗口，直接保存图片
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ================== 任务1：数据准备 ==================
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

print("===== 数据集信息 =====")
print(f"总样本数：{len(images)}")
print(f"图像尺寸：{images.shape[1]}×{images.shape[2]}")
print(f"特征向量维度：{X.shape[1]}")
print(f"类别：{np.unique(y)}")

# 保存样本图片
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"{y[i]}")
    plt.axis('off')
plt.suptitle("Sample Images")
plt.savefig("01_sample_images.png", dpi=150)
plt.close()

# ================== 任务2：数据划分 ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\n训练集数量：{len(X_train)}")
print(f"测试集数量：{len(X_test)}")

# ================== 任务4：模型训练与评估 ==================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
y_pred_dict = {}

print("\n===== 各模型测试准确率 =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    y_pred_dict[name] = y_pred
    print(f"{name:<20}: {acc:.4f}")

# ================== 任务5：结果输出 ==================
print("\n===== 准确率表格 =====")
for k,v in results.items():
    print(f"| {k:15} | {v:.4f} |")

# ================== 任务6：错误样本分析（SVM） ==================
best_model = "SVM"
y_pred = y_pred_dict[best_model]
cm = confusion_matrix(y_test, y_pred)

# 保存混淆矩阵
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"{best_model} Confusion Matrix")
plt.xlabel("Pred")
plt.ylabel("True")
plt.savefig("02_confusion_matrix.png", dpi=150)
plt.close()

# 保存错误分类样本
err_idx = np.where(y_pred != y_test)[0]
plt.figure(figsize=(10,5))
for i, idx in enumerate(err_idx[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    plt.title(f"True:{y_test[idx]}\nPred:{y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Misclassified Samples")
plt.savefig("03_misclassified.png", dpi=150)
plt.close()

print("\n✅ 代码运行完成！所有图片已自动保存！")