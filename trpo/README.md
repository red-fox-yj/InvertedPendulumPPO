# TRPO
TRPO旨在求解一个带有约束的优化问题。我们考虑旧策略 $\pi_{\text{old}}$ 和新策略 $\pi_\theta$ ，目标是提高策略的性能，同时限制新旧策略之间的差异。

## 基本优化问题

在强化学习中，我们常希望最大化策略的期望回报。对于策略梯度方法，一个经典的无约束优化目标可以写为（以优势函数 $A^{\pi_{\text{old}}}(s,a)$ 为例）：  
$$
\max_{\theta} \; \mathbb{E}_{s,a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s,a) \right].
$$

这里，$\pi_\theta(a|s)$ 表示在状态 $s$ 下由参数 $\theta$ 描述的策略选择动作 $a$ 的概率，$A^{\pi_{\text{old}}}(s,a)$ 是在旧策略下定义的优势函数，用来衡量在该状态-动作对下，采取该动作的相对好坏。

然而，直接最大化上述期望可能导致策略更新过大，从而破坏先前已经学到的行为模式并导致不稳定。为了解决这个问题，TRPO在优化中加入了一个约束，限制新旧策略之间的差异。

## 约束条件（KL散度）

TRPO使用KL散度来度量新旧策略之间的差异性。对于给定状态 $s$，两个分布 $\pi_{\text{old}}(\cdot|s)$ 和 $\pi_\theta(\cdot|s)$ 之间的KL散度定义为：
$$
D_{\mathrm{KL}}\bigl(\pi_{\text{old}}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)\bigr) = \sum_{a} \pi_{\text{old}}(a|s) \log \frac{\pi_{\text{old}}(a|s)}{\pi_{\theta}(a|s)}.
$$

TRPO对KL散度在状态分布上的期望加以约束：
$$
\mathbb{E}_{s \sim \pi_{\text{old}}}\bigl[D_{\mathrm{KL}}(\pi_{\text{old}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s))\bigr] \leq \delta,
$$
其中 $\delta$ 是一个预先设定的小正数，用来限制新策略与旧策略的差异程度，从而保持更新的稳定性。

## 最终的优化问题

将目标和约束放在一起，我们有：
$$
\max_{\theta} \; \mathbb{E}_{s,a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s,a) \right]
$$
subject to
$$
\mathbb{E}_{s \sim \pi_{\text{old}}}\bigl[D_{\mathrm{KL}}(\pi_{\text{old}}(\cdot|s)\,\|\,\pi_\theta(\cdot|s))\bigr] \leq \delta.
$$

直接求解上述约束优化问题非常困难。TRPO通过对目标函数和约束进行二阶近似来简化求解：

1. **对KL散度的二阶近似**：  
   在旧参数 $\theta_{\text{old}}$ 附近，对KL散度进行泰勒展开，利用Fisher信息矩阵（$F$）作为二阶近似：
   $$
   \mathbb{E}_{s \sim \pi_{\text{old}}}\bigl[D_{\mathrm{KL}}(\pi_{\text{old}}\|\pi_\theta)\bigr] \approx \frac{1}{2} (\theta - \theta_{\text{old}})^\top F (\theta - \theta_{\text{old}})
   $$

   这里，Fisher信息矩阵 $F$ 通常定义为：
   $$
   F = \mathbb{E}_{s \sim \pi_{\text{old}}}\left[ \mathbb{E}_{a \sim \pi_{\text{old}}(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top \right] \right]_{\theta=\theta_{\text{old}}}.
   $$

2. **对目标函数的线性近似**：  
   将新的目标函数在 $\theta_{\text{old}}$ 附近做一阶泰勒展开：
   $$
   L(\theta) \approx L(\theta_{\text{old}}) + g^\top (\theta - \theta_{\text{old}}),
   $$
   其中
   $$
   g = \nabla_\theta \mathbb{E}_{s,a \sim \pi_{\text{old}}} \left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} A^{\pi_{\text{old}}}(s,a)\right]_{\theta=\theta_{\text{old}}}.
   $$

这样，原问题化简为：
$$
\max_{\theta} \; g^\top (\theta - \theta_{\text{old}})
$$
subject to
$$
\frac{1}{2} (\theta - \theta_{\text{old}})^\top F (\theta - \theta_{\text{old}}) \leq \delta.
$$

这个问题的解在数学上相当于沿自然梯度方向进行有界步长的更新。通过求解此二次规划问题（通常用共轭梯度法来高效近似地求解），可以得到更新方向 $\Delta \theta = \theta - \theta_{\text{old}}$。

## 结论

从公式的角度，总结一下TRPO的核心：

- 原始目标：最大化策略性能的线性近似项。
- 约束：通过KL散度的二阶近似形成二次形式约束。
- 求解得到的更新方向与自然梯度方向相关，从而实现稳定且受控制的策略更新。

这种公式化的处理方法使得TRPO在策略迭代的优化过程中能像“信任域优化”一样，以较为保守且稳定的方式对策略进行改进。