import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pre_processing import get_data

# 示例数据，假设有5个样本，每个样本有4个特征
data, _, label_m, label_m_no_nor, label_q_nor, __, X = get_data(4)

# 使用sklearn的PCA进行计算
pca = PCA()
X_pca = pca.fit_transform(X)

# 获取sklearn计算的特征值（解释方差）
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
components = pca.components_

# 设置阈值，比如选择累计解释方差比例达到90%
threshold = 0.90
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
num_components_to_retain = np.argmax(cumulative_variance_ratio >= threshold) + 1

print(f'Number of components to retain {threshold * 100}% variance: {num_components_to_retain}')

# 绘制投影后的数据和主成分
num_components = X_pca.shape[1]

plt.figure(figsize=(15, 10))

# 绘制每个主成分的投影图
for i in range(num_components):
    plt.subplot(2, num_components, i + 1)
    plt.plot(X_pca[:, i], color='blue', marker='o', linestyle='None')
    plt.xlabel(f'Principal Component {i + 1}')
    plt.ylim(-0.1, 0.1)  # 设置y轴范围
    plt.title(f'PC {i + 1} Projection')

    plt.subplot(2, num_components, num_components + i + 1)
    plt.bar(range(1, X.shape[1] + 1), components[i], alpha=0.7, color='orange')
    plt.xlabel('Original Features')
    plt.ylabel('Component Weights')
    plt.title(f'PC {i + 1} Weights')

plt.tight_layout()
plt.show()

# 绘制解释方差比例
plt.figure(figsize=(8, 4))
plt.bar(range(1, num_components + 1), explained_variance_ratio, alpha=0.7, color='green')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained Ratio')
plt.title('Explained Variance Ratio by Principal Components')

# 绘制累计解释方差比例
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', color='blue', label='Cumulative Explained Variance')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'{threshold * 100}% Threshold')
plt.legend()
plt.xticks(range(1, num_components + 1))
plt.show()

a = 0.95
b = 1 - a

comp_pca = (a*X_pca[:, 0]) + (b * X_pca[:, 1])
plt.plot(comp_pca)
plt.show()