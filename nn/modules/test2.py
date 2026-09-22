import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from tqdm import tqdm

# ==========================
# 1. 读取网络
# ==========================
G = nx.read_edgelist(
     r"D:\dataset\email-Eu-core.txt\email-Eu-core.txt",
    nodetype=int
)

nodes = list(G.nodes())
n = len(nodes)

print(f"Nodes: {n}")
print(f"Edges: {G.number_of_edges()}")

node_to_idx = {node: idx for idx, node in enumerate(nodes)}

# ==========================
# 2. Jaccard Similarity
# ==========================
print("Computing Jaccard...")

jaccard_matrix = np.zeros((n, n))

for i in range(n):
    jaccard_matrix[i, i] = 1

for u, v in tqdm(combinations(nodes, 2), total=n*(n-1)//2):

    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))

    union = len(Nu | Nv)

    if union == 0:
        sim = 0
    else:
        sim = len(Nu & Nv) / union

    i = node_to_idx[u]
    j = node_to_idx[v]

    jaccard_matrix[i, j] = sim
    jaccard_matrix[j, i] = sim

print("Jaccard Done")


# ==========================
# 3. Cosine Similarity
# ==========================
print("Computing Cosine...")

A = nx.to_numpy_array(G, nodelist=nodes)

cosine_matrix = cosine_similarity(A)

print("Cosine Done")


# ==========================
# 4. SimRank Similarity
# ==========================
print("Computing SimRank...")

C = 0.8
max_iter = 10

simrank_matrix = np.eye(n)

for iteration in range(max_iter):

    print(f"Iteration {iteration+1}/{max_iter}")

    new_simrank = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):

            u = nodes[i]
            v = nodes[j]

            Nu = list(G.neighbors(u))
            Nv = list(G.neighbors(v))

            if len(Nu) == 0 or len(Nv) == 0:
                continue

            s = 0

            for nu in Nu:
                for nv in Nv:

                    s += simrank_matrix[
                        node_to_idx[nu],
                        node_to_idx[nv]
                    ]

            s *= C / (len(Nu) * len(Nv))

            new_simrank[i, j] = s
            new_simrank[j, i] = s

    simrank_matrix = new_simrank

print("SimRank Done")


# ==========================
# 5. 保存结果
# ==========================
np.save("jaccard_matrix.npy", jaccard_matrix)
np.save("cosine_matrix.npy", cosine_matrix)
np.save("simrank_matrix.npy", simrank_matrix)

print("Saved:")
print("jaccard_matrix.npy")
print("cosine_matrix.npy")
print("simrank_matrix.npy")


# ==========================
# 6. 查看示例
# ==========================
print("\nExample:")

u = nodes[0]
v = nodes[1]

i = node_to_idx[u]
j = node_to_idx[v]

print(f"Node Pair: ({u},{v})")

print("Jaccard :", jaccard_matrix[i, j])
print("Cosine  :", cosine_matrix[i, j])
print("SimRank :", simrank_matrix[i, j])


import pandas as pd

pd.DataFrame(
    jaccard_matrix,
    index=nodes,
    columns=nodes
).to_csv("jaccard_matrix.csv")

pd.DataFrame(
    cosine_matrix,
    index=nodes,
    columns=nodes
).to_csv("cosine_matrix.csv")

pd.DataFrame(
    simrank_matrix,
    index=nodes,
    columns=nodes
).to_csv("simrank_matrix.csv")