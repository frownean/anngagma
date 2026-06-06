# AGMA 多模态融合文献复核与改进方案

本文档是在登录 IEEE Xplore 并通过 Keio University 机构权限访问后重新整理的版本。文献主要来自 IEEE Xplore 中 `multimodal fusion HAR`、`multimodal fusion human activity recognition` 及相关 HAR 多模态融合关键词的检索结果。

## 对核心假设的客观判断

你的研究动机在技术上是合理的，但论文表述需要更严谨。建议不要直接写成“WiFi 的头部语义和 mmWave 的腿部语义被错误融合”，除非模型中有明确的人体部位标签、可视化证据或语义监督。更稳妥的表述是：

朴素多模态融合可能会混合不同模态中并不对应的潜在动作因素、身体部位相关证据、动作阶段或模态特有线索，从而造成跨模态语义槽错配，损害原本有判别力的模态信息。

结合当前代码，AGMA 更适合被描述为：

AGMA 学习一组共享的潜在语义锚点，并将其作为跨模态语义槽。每个模态先把自身 token 映射到同一组 anchor 上，再在对应 anchor 级别进行融合。这样做的目标是在融合前尽量减少跨模态语义槽错配。

当前进一步实现的核心模块是 **Reference-Guided Positive Anchor Mining**：以 mmWave 作为强参考模态，在每个样本和每个 anchor 上筛选 WiFi/RFID 是否提供正向证据。若弱模态 anchor 与 mmWave anchor 一致，则增强其融合贡献；若不一致，则在融合 gate、alignment loss 和 auxiliary loss 中软抑制其影响。

当前“三模态融合性能低于单模态 mmWave”的现象并不罕见，也不一定说明研究方向错误。它通常说明以下问题之一：

- 弱模态被过度赋权；
- 对齐约束过强，反而把强模态 mmWave 拉向弱模态；
- 对比学习正样本定义过宽，导致同类不同动作阶段或不同身体部位线索被强行拉近；
- 融合模块缺少可靠性建模，不能判断何时应该相信某个模态。

因此，顶刊级方法不能只强调“多模态融合”，而应证明融合是可靠性感知的，并且不会在没有证据时破坏最强模态的信息。

## 复核的 10 篇 IEEE Xplore 相关论文

1. [Graph Convolutional Network-Based Multimodal Uncertainty Fusion for Human Activity Recognition](https://ieeexplore.ieee.org/document/11314519)，IEEE Internet of Things Journal，2026。
   核心思想：多模态 HAR 需要在融合前显式建模不确定性。对 AGMA 的启发是：融合 gate 不应只表示语义一致性，还应表示模态可靠性。因此当前代码加入可靠性先验 gate 是合理的，也需要在实验中报告每个模态的 gate 权重统计。

2. [MSMFT: Multi-Stream Multimodal Factorized Transformer for Human Activity Recognition](https://ieeexplore.ieee.org/document/10850630)，IEEE Sensors Journal，2025。
   核心思想：多流结构和 factorized transformer 可以分别建模模态交互与时序交互。对 AGMA 的启发是：anchor 级融合可以作为 dense cross-modal attention 的替代方案，但必须通过 anchor 诊断证明这些锚点确实具有稳定的语义槽作用。

3. [Human Action Recognition Using Deep Multilevel Multimodal Fusion of Depth and Inertial Sensors](https://ieeexplore.ieee.org/document/8869853)，IEEE Sensors Journal，2020。
   核心思想：不同层级的多模态融合可能比单一融合点更有效。对 AGMA 的启发是：当前代码主要在 anchor 聚合后融合，后续可考虑浅层/中层辅助对齐或分阶段融合消融。

4. [Virtual Fusion With Contrastive Learning for Single-Sensor-Based Activity Recognition](https://ieeexplore.ieee.org/document/10559768)，IEEE Sensors Journal，2024。
   核心思想：对比学习可以增强传感器表征，并模拟或补充多传感器信息。对 AGMA 的启发是：anchor 级对比对齐是合理方向，但正样本定义必须谨慎。当前代码默认只使用同一样本的跨模态 anchor 作为正样本，而不是把所有同类样本都拉近。

5. [In-Home Human Activity Recognition via Kinematics-Focused Multimodal Sensor Fusion and Spatio-Temporal Neural Architecture](https://ieeexplore.ieee.org/document/11192600)，IEEE Sensors Letters，2025。
   核心思想：面向家庭场景的 HAR 需要关注运动学信息和时空结构建模。对 AGMA 的启发是：需要通过时序和 anchor 诊断证明 anchor 不只是任意池化特征，而是捕获了动作阶段或运动组成部分。

6. [PWLF: A Robust Weighted Late Multimodal Fusion Approach](https://ieeexplore.ieee.org/document/11412698)，IEEE 会议论文，2025。
   核心思想：当不同模态可靠性不同，带权 late fusion 是一个稳健基线。对 AGMA 的启发是：你的方法不仅要超过 concat 或平均融合，还应与加权 late fusion 这类更强鲁棒基线比较。

7. [Research on Action Recognition Algorithm Based on Multimodal Data Fusion](https://ieeexplore.ieee.org/document/11084160)，IEEE 会议论文，2025。
   核心思想：多模态动作识别通常将不同特征提取器的输出组合后分类。对 AGMA 的启发是：这类方法更像常规特征融合基线，AGMA 的创新性应重点对比 attention、uncertainty fusion、transformer fusion 等更强方法。

8. [Multimodal Fusion of Wearable and Vision Data: Exploring Early and Model Fusion for Human Activity Understanding](https://ieeexplore.ieee.org/document/11167801)，IEEE 会议论文，2025。
   核心思想：early fusion 和 model-level fusion 有不同优缺点。对 AGMA 的启发是：实验中应包含 early/concat、late fusion 和 AGMA，在相同 subject split 下证明语义锚点融合的优势。

9. [Evaluating Outputs Fusion Technique in Multimodal Human Activity Recognition: Impact of Modality Reduction on Performance Efficiency](https://ieeexplore.ieee.org/document/10812599)，IEEE 会议论文，2024。
   核心思想：输出级融合和模态数量变化会影响性能与效率。对 AGMA 的启发是：缺失模态评估不是附加实验，而是证明鲁棒性和实用价值的核心实验。

10. [Deep Feature Fusion-Based Human Activity Recognition from Multimodal Sensor Data](https://ieeexplore.ieee.org/document/11291292)，IEEE 会议论文，2025。
    核心思想：深度特征融合是多模态传感器 HAR 中常见基线。对 AGMA 的启发是：论文需要明确说明 anchor 级语义对齐为什么比普通深度特征拼接更有原则性。

## 原始 AGMA 代码中发现的问题

- 原融合 gate 奖励的是“与所有模态平均共识的一致性”。如果 WiFi/RFID 噪声较强，这种平均共识可能会把 mmWave 从原本更有判别力的表示拉偏。
- `modality_prior` 原本只用于 gate balance 正则项，并没有直接影响 gate logits，因此对实际融合权重的控制较弱。
- 动态 anchor 的 sample context 原本是所有模态 summary 的简单平均，因此弱模态可能在 anchor 生成阶段就污染语义锚点。
- `gate_floor` 和 `gate_balance_weight` 可能强制弱模态进入融合表示。当单模态 mmWave 已经最强时，这种约束存在风险。
- `assignment_consistency_weight` 原本使用 detached assignment strength，因此无法真正优化语义槽一致性。
- 对比学习中如果把所有同类样本都作为正样本，可能过于激进。不同 trial 的同一动作可能包含不同动作阶段或身体部位线索，强行拉近会模糊语义槽。

## 已实现的代码改进

- 加入 `modality_prior_logit_weight`，让模态可靠性先验直接作用于 gate logits，而不是只作为弱正则项。
- 加入 `context_prior_weight`，让动态 anchor 生成过程也具备可靠性感知能力，避免简单平均所有模态。
- 加入 `reference_modality`、`reference_agreement_weight` 和 `reference_residual_weight`，默认以 mmWave 作为 reference modality，为融合结果保留一条回到最强模态的受控路径。
- 加入 `positive_anchor_mining`，实现以 mmWave 为参考的 anchor 级正向证据挖掘。其核心权重为：

```text
agreement(m, j) = cosine(z_m,j, stopgrad(z_mmwave,j))
positive_weight(m, j) = sigmoid((agreement(m, j) - threshold) / temperature)
```

- `positive_weight` 被用于三处：作为 weak modality 的 fusion gate bias、作为 alignment loss 的软降权因子、作为 auxiliary loss 的样本级软降权因子。
- 修复 assignment strength 的梯度问题，使 assignment consistency 能真正参与优化；但在 alignment loss 中仍将其作为置信度权重 detach，避免不稳定反馈。
- 启用轻量训练期 anchor auxiliary heads，用于保持各单模态尤其是 mmWave 的判别信息；这些辅助头只在训练时使用，不增加推理开销。
- 加入 `contrast_same_class_positives`，默认设为 `false`，即 anchor 对比只使用同一样本跨模态正样本；是否使用同类正样本作为单独消融。
- 增加消融项：`no_positive_anchor_mining`、`no_positive_gate_bias`、`no_positive_alignment_weighting`、`no_positive_aux_weighting`、`no_reliability_prior`、`no_context_prior`、`no_reference_residual`、`no_anchor_auxiliary`、`class_positive_contrast`。

## 推荐实验表

建议在完全相同的数据划分下运行以下实验：

- 单模态：mmwave、wifi、rfid；
- concat_fusion；
- late_fusion；
- AGMA full；
- AGMA + positive_anchor_mining；
- AGMA no_positive_anchor_mining；
- AGMA no_positive_gate_bias；
- AGMA no_positive_alignment_weighting；
- AGMA no_positive_aux_weighting；
- AGMA no_alignment；
- AGMA no_reliability_prior；
- AGMA no_context_prior；
- AGMA no_reference_residual；
- AGMA no_anchor_auxiliary；
- AGMA class_positive_contrast；
- 不同 `reference_residual_weight`：0.0、0.1、0.2、0.3；
- 不同 `modality_prior_logit_weight`：0.0、0.2、0.35、0.5；
- 不同 positive mining threshold：0.00、0.10、0.15、0.20、0.30；
- 不同 positive mining temperature：0.05、0.10、0.20；
- 所有多模态模型的缺失模态评估。

这些实验应至少报告：

- accuracy；
- macro F1；
- balanced accuracy；
- per-subject 结果；
- per-class 结果；
- 模态 gate 平均权重；
- anchor assignment entropy；
- anchor-level positive weight 分布；
- WiFi/RFID 被 positive mining 增强的 anchor 比例；
- 缺失模态场景下的性能变化。

## 论文定位建议

仅仅写“multimodal fusion”不够强，因为 IEEE Xplore 中已经有大量类似工作。更有创新性的表述可以是：

面向异构 RF 人体活动识别的可靠性感知 anchor 级正向证据挖掘与语义对齐融合方法。

英文可写为：

Reference-guided positive anchor mining for reliability-aware heterogeneous RF-based human activity recognition.

为了让顶刊审稿人相信方法合理，需要补充以下证据：

- full AGMA 能超过或至少接近单模态 mmWave，同时在缺失模态或噪声模态场景下更鲁棒；
- gate 分析显示：当 WiFi/RFID 不可靠时，模型会更依赖 mmWave；但在特定类别或特定 anchor 上，WiFi/RFID 仍能提供有用补充；
- positive anchor mining 分析显示：WiFi/RFID 并非整体被丢弃，而是在与 mmWave 一致的 anchor 上被选择性增强；
- anchor 诊断显示：不同 anchor 不是完全坍缩的，而是承担不同动作阶段、局部运动或判别线索；
- 缺失模态和噪声模态实验能够证明方法不是简单依赖 mmWave；
- positive mining、alignment、contrast、reliability gate、context prior、reference residual、anchor auxiliary 和 anchor 数量都有清晰消融。

如果最终 full AGMA 仍然不能超过单模态 mmWave，应诚实地调整论文主张：不要声称“融合提升最高准确率”，而应改为“在保持强模态性能的同时提升鲁棒性、校准性和缺失模态容忍能力”。
