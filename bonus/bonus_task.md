# 种子选择与区域聚合

INSID3 用去偏特征做跨图匹配，用原始特征计算图内相似度。本题输入包含这两套特征、聚类标签和候选 patch 掩码。请实现种子选择与区域聚合，无需做特征提取、SVD、聚类或候选定位。

## 参考资料

- [论文 §3.3–3.4，式 (9)–(14)](https://arxiv.org/abs/2603.28480)。
- [作者 `_seed_and_aggregate`, L311–364](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L311-L364)。
- [作者 `compute_cluster_prototypes`, L28–44](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/clustering.py#L28-L44)。
- [调用处，L174–186](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L174-L186)。

请按作者 commit `0c165a10cf52ab91f335883d06260de86854adbe` 实现。论文与该版本在最终掩码计算上有差别；聚合评分、覆盖比例、种子权重和阈值比较均以链接中的代码为准。

## 函数接口

新建 `src/insid3_bonus.py`，仅依赖 NumPy，实现：

```python
def select_seed_and_merge(
    reference_prototype: np.ndarray,  # (C,)
    target_original: np.ndarray,      # (H, W, C)
    target_debiased: np.ndarray,      # (H, W, C)
    cluster_labels: np.ndarray,       # (H, W)
    candidate_mask: np.ndarray,       # (H, W)
    threshold: float,
) -> dict:
    ...
```

测试输入满足以下条件，无需另写非法输入处理：

- 所有特征有限、实数、通道相同，两个目标网格完全对齐。
- `target_original` 和 `target_debiased` 的非零 patch 已分别 L2 归一化；前者是去偏前的特征，后者是去偏并重新归一化后的特征。零 patch 保持零。
- `reference_prototype` 是去偏空间中的单位向量。
- 标签是连续整数 `0, …, K-1`，每个标签都至少出现一次，没有 `-1` 标签。
- `candidate_mask` 是已经算好的布尔 patch 掩码，标记的是 patch，候选 cluster 需要据此确定。
- `threshold` 是有限实数。输入数组不得修改。

函数需要确定候选 cluster、构造 cluster 原型、选择种子，再计算所有 cluster 的聚合分数和最终掩码。原型归一化使用 `x / max(||x||₂, 1e-12)`；均值为零时返回零向量。

返回字典包含：

| 键 | 约定 |
| --- | --- |
| `seed_id` | 选中的实际 cluster 标签，Python `int`；候选集合为空时为 `None`。种子评分完全相同时，取标签最小者。 |
| `cross_similarity` | `(K,)`，作者代码**聚合阶段**的逐 cluster 跨图分数。 |
| `intra_similarity` | `(K,)`，逐 cluster 与种子的图内相似度。 |
| `area_weights` | `(K,)`，执行作者代码中的种子权重处理之后的覆盖比例。 |
| `combined_scores` | `(K,)`，阈值判断前的最终分数。 |
| `final_mask` | `(H,W)` 布尔数组，按固定版本作者代码的阈值规则恢复到完整 cluster。 |

数组的第 `k` 项始终对应标签 `k`。若没有任何候选 patch，返回 `seed_id=None`、四个长度为 `K` 的零数组和全 False 掩码。这是对作者提前返回分支的接口补充。

## 运行测试

从仓库根目录运行：

```bash
python -m pytest -q bonus/test_bonus_task.py
```

请提交函数代码及简短说明，解释选择种子和聚合区域时各自使用的特征与评分方式。无需实现完整 INSID3。
