import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ==================================================
# 修改为你的 SNAP 数据文件路径
# ==================================================
file_path = r"CA-GrQc.txt"

# ==================================================
# 读取网络
# SNAP数据一般以#开头为注释
# ==================================================
print("正在读取网络...")

G = nx.read_edgelist(
    file_path,
    comments='#',
    nodetype=int
)

print("读取完成！")

# ==================================================
# 基本统计
# ==================================================
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("\n========== 网络基本信息 ==========")
print(f"节点数: {num_nodes}")
print(f"边数: {num_edges}")

# ==================================================
# 计算度
# ==================================================
degrees = [d for _, d in G.degree()]

avg_degree = np.mean(degrees)
max_degree = np.max(degrees)
min_degree = np.min(degrees)
std_degree = np.std(degrees)

print("\n========== 度统计 ==========")
print(f"平均度: {avg_degree:.2f}")
print(f"最大度: {max_degree}")
print(f"最小度: {min_degree}")
print(f"度标准差: {std_degree:.2f}")

# ==================================================
# 统计度分布
# ==================================================
degree_count = Counter(degrees)

x = sorted(degree_count.keys())
y = [degree_count[d] for d in x]

# ==================================================
# 绘制折线图
# ==================================================
plt.figure(figsize=(8, 6))

plt.plot(
    x,
    y,
    marker='o',
    linewidth=1.5
)

plt.title("Degree Distribution (Line)")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.grid(True)

plt.tight_layout()
plt.savefig("degree_distribution_line.png", dpi=300)

print("\n折线图已保存:")
print("degree_distribution_line.png")

# ==================================================
# 绘制散点图
# ==================================================
plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    y,
    s=20
)

plt.title("Degree Distribution (Scatter)")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.grid(True)

plt.tight_layout()
plt.savefig("degree_distribution_scatter.png", dpi=300)

print("散点图已保存:")
print("degree_distribution_scatter.png")

# ==================================================
# 绘制Log-Log图
# ==================================================
plt.figure(figsize=(8, 6))

plt.scatter(
    x,
    y,
    s=20
)

plt.xscale("log")
plt.yscale("log")

plt.title("Degree Distribution (Log-Log)")
plt.xlabel("Degree (log)")
plt.ylabel("Frequency (log)")
plt.grid(True)

plt.tight_layout()
plt.savefig("degree_distribution_loglog.png", dpi=300)

print("Log-Log图已保存:")
print("degree_distribution_loglog.png")

# ==================================================
# 显示全部图像
# ==================================================
plt.show()

print("\n分析完成！")