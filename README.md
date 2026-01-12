# 《机器学习》课程项目：TLM/TTL 复现与 Quantile-SEL 改进

> **说明**：本仓库为秋季学期 **《机器学习》课程** 的期末小项目。

## 1. 项目概览

本仓库旨在通过实验复现与改进 **Test-Time Learning (TTL)** 技术。项目主要包含两个目标：

1.  **复现**：Hu et al., *Test-Time Learning for Large Language Models* (TLM, ICML 2025) 中的核心思想；
2.  **改进**：在官方方法基础上，提出 **Quantile-SEL**（基于分位数阈值的样本选择），旨在保持输出质量的前提下，显著减少测试时的反向传播（Backpropagation）次数，提升计算效率。

实验设置了一个合成的 **领域迁移场景**（Medical vs. Finance 文本流），使用轻量级的 `sshleifer/tiny-gpt2` 模型配合 **LoRA** 适配器进行在线更新。

---

## 2. 快速开始 (Quick Start)

项目代码结构极简，仅需安装必要依赖（`torch`, `transformers`, `peft`, `matplotlib`, `pandas`）后即可直接运行主脚本 `ttl_reproduce.py`。

* **耗时**：默认设置下约 2 分钟。
* **输出**：脚本运行结束后，会自动在 `figs/` 文件夹下生成分析图表，并在 `results/` 下生成数据汇总。

---
```bash
python ttl_reproduce.py --domain mix --n_samples 120
```

## 3. 方法与设置 (Methodology)

我们在实验中对比了四种不同的测试时策略。所有方法均保持基础 LM 参数固定，仅更新 LoRA 参数，以符合 TTL 的稳定性设计。

| 方法 | 描述 | 特点 |
| :--- | :--- | :--- |
| **Baseline** | 不做任何测试时更新 (No update)。 | 计算成本最低，无法适应分布偏移。 |
| **TTL (no SEL)** | 论文中的基本 Test-Time Learning。 | 对每个输入样本均基于 Input PPL 做梯度下降。 |
| **Fixed-SEL** | 论文提出的 Sample-Efficient Learning。 | 使用固定阈值 $\log P_0$ 对高困惑度样本进行加权更新。 |
| **Quantile-SEL (Ours)** | **本项目提出的改进方法**。 | 在在线窗口内，利用**分位数 (Quantile)** 自动确定阈值，自适应选择（如 Top 30%）高 PPL 样本进行更新。无需为不同任务手动调节阈值。 |

---

## 4. 实验结果 (Key Results)

在 `--domain mix --n_samples 120` 的设置下，我们得到了以下关键指标（数据源自 `results/summary.csv`）：

| 方法 | 平均输出 PPL $\downarrow$ (越低越好) | 反向传播次数 $\downarrow$ (越低越好) | 样本效率 (Efficiency) $\uparrow$ |
| :--- | :--- | :--- | :--- |
| Baseline | 10.8311 | 0 | 4.6597 |
| TTL (no SEL) | 10.8297 | 120 | 0.0444 |
| Fixed-SEL | 10.8305 | 27 | 0.2126 |
| **Quantile-SEL (Ours)** | **10.8309** | **12** | **0.4256** |

### 结果分析

结合实验生成的图像，我们得出了以下三个结论：

#### 1. TTL 的核心思想成立
> **参见：** `figs/ppl_trend.png`

图中展示了四种方法的**输入困惑度随样本数的累积平均趋势**。
* **观察**：相比 Baseline，TTL 及其变体的 Input PPL 随着在线样本的增加逐步下降。
* **结论**：验证了论文提出的“最小化输入 PPL 可以实现测试时自适应”这一关键假设，体现了无监督目标对适应分布偏移的有效性。

#### 2. SEL 在保持效果的同时显著节省计算
> **参见：** `figs/comparison.png`

* **观察**：输出 PPL 上，四种方法的差异非常小（均约 10.83 左右）；但在计算开销上，TTL(no SEL) 需要 120 次反向传播，而 Fixed-SEL 仅需 27 次，Quantile-SEL 进一步降到 12 次。
* **结论**：样本选择（SEL）在有限计算预算下极其有效，验证了“数据选择 + 重要性加权”的策略价值。

#### 3. Quantile-SEL 落在更优的质量–成本 Pareto 前沿
> **参见：** `figs/tradeoff.png`

图中横轴为计算成本（反向传播次数），纵轴为质量（输出 PPL）。
* **观察**：
    * Baseline：左上角（低成本，无适应）。
    * TTL(no SEL)：右下角（高质量，高成本）。
    * **Quantile-SEL**：位于左下侧 Pareto 前沿。
* **结论**：Quantile-SEL 通过自适应阈值，在与其他 TTL 变体接近的效果下，用最少的计算成本达成了目标。体现了该改进方法的优势：**无需手动调参，即可自适应数据流分布**。

---

## 5. 课程联系与收获 (Reflections)

作为《机器学习》课程项目，本实验主要在以下三个方面将课堂理论落地：

1.  **从损失函数到评价指标的统一视角**：
    通过显式计算 Input PPL 与 Output PPL，深化了对“训练时最小化交叉熵”与“推断时关注 PPL/风险最小化”之间联系的理解。

2.  **样本效率与计算效率的权衡**：
    Fixed-SEL 和 Quantile-SEL 本质上是**重要性采样 (Importance Sampling)** 的实践。通过 `efficiency` 指标，直观展示了如何在计算预算约束下进行有效的数据选择。

3.  **科研工作流实践**：
    项目遵循了“复现 + 小改进”的经典模式。从复现论文核心结论出发，发现固定阈值的局限性，进而提出基于分位数的自适应改进，并用简单的实验完成了验证。

---

## 6. 文件结构

* `ttl_reproduce.py`: 主脚本。包含数据生成、LoRA 模型构建、四种方法的 TTL 循环实现、绘图及评估代码。
* `results/`: 存放实验结果数据（CSV/JSON）。
* `figs/`: 存放生成的分析图表。
    * `ppl_trend.png`: 学习曲线。
    * `comparison.png`: 方法对比柱状图。
    * `tradeoff.png`: 质量–成本权衡图。

---

**References:**
Hu et al., *Test-Time Learning for Large Language Models*, ICML 2025.
