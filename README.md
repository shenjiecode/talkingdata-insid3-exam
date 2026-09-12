# INSID3：位置去偏后的跨图匹配

使用 NumPy 处理已提取的特征，在 CPU 上运行，无需 DINOv3 权重或外部数据集。

## 论文与代码定位

| 内容 | 说明 |
| --- | --- |
| 论文依据 | [INSID3 §3.1](https://arxiv.org/abs/2603.28480)：式 (3) 后通过 SVD 估计位置子空间；式 (4) 对参考与目标特征去偏；式 (2) 从参考掩码区域计算原型，再与目标 patch 比较。 |
| 作者代码依据 | `models/insid3.py` 中，[`_build_positional_basis`, L246–261](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L246-L261) 归一化、中心化 probe 并提取位置基；[`_debias_features`, L263–271](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L263-L271) 投影去偏后重新归一化；[`predict_mask`, L136–158](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L136-L158) 归一化输入 patch 并计算参考原型；[`_locate_candidates`, L285–286](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L285-L286) 计算跨图点积。 |
| 我的理解 | 不同图像中相同坐标的特征可能因位置编码而相似。低语义 probe 用来估计这些方向，再对参考与目标使用同一个正交补投影，抑制跨图比较中的位置干扰。参考掩码决定要匹配的区域。 |

源码链接固定到 commit `0c165a10cf52ab91f335883d06260de86854adbe`。本仓库先提交论文与代码定位，再实现函数，最后编写并运行测试。

## 实现说明

函数位于 `src/insid3_matching.py`。probe 逐 patch 做 L2 归一化，展开成 `(P,C)`，按通道中心化，再取 SVD 的前 `r` 个右奇异向量。作者代码使用 `(C,P)` 排列，因此对应左奇异向量。

参考与目标 patch 先归一化，再计算 `X - (X B) Bᵀ` 并重新归一化。只对 probe 做中心化。参考掩码内的去偏 patch 求均值后再次归一化，与目标去偏特征点积，返回 `basis`、`reference_prototype` 和 `similarity_map`。

- `rank` 为非负整数，`r = min(rank, Hp*Wp, C)`；`rank=0` 不去除方向。不按数值秩截断，秩亏时零奇异值对应方向不唯一。
- 归一化使用 `x / max(||x||₂, 1e-12)`，零向量保持零。计算使用 float64，不修改输入。
- 三个网格可为不同的非方形尺寸，但通道数必须一致。空参考掩码、形状不匹配、非二值掩码、非有限特征或非法 `rank` 均明确报错。

## 测试

要求 Python 3.10 或更高版本。在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

实际结果：`43 passed in 0.12s`。验证环境为 Windows、Python 3.12.8、NumPy 2.5.3、pytest 9.1.1。

`tests/test_insid3_matching.py` 覆盖位置基的维度与正交性、归一化顺序、位置与语义冲突、掩码变化、`rank=0`、非方形网格、空掩码、非法输入及输入不被修改。
合成例子中，“语义相同、位置相反”的得分从 `-99/101` 变为 `1`，“位置相同、语义无关”的得分从 `100/101` 变为 `0`。

## 加分项

第二题见 [题面](bonus/bonus_task.md)、[设计说明](bonus/bonus_design.md) 和 `bonus/test_bonus_task.py`。将待测模型答案放入 `src/insid3_bonus.py` 后，运行 `python -m pytest -q bonus/test_bonus_task.py`。默认测试命令只运行第一题。

第三题见 [审核结论](manual_review_1_1_1.md)。

## AI 使用说明

AI 辅助完成论文与代码分析、实现、测试、题目设计及审核文本，人工核验待完成。
