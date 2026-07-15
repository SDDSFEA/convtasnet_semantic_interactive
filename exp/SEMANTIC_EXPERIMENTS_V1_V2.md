# Conv-TasNet Semantic Negotiation：V1/V2 实验记录

本文记录 `exp/librimix_clean_semantic` 和 `exp/librimix_clean_semantic_v2` 两次实验使用的代码、数据、训练设置、关键结果与已观察到的现象。

## 1. 共同实验设置

两次实验都在两说话人 Libri2Mix clean offset 数据上进行，使用完整 utterance，而不是随机 4 秒切块。这样可以避免把完整 utterance 的 ASR semantic hidden states 提供给局部音频 chunk 所造成的上下文泄漏。

共同模型规模：

```text
N = 512
L = 16
B = 128
H = 512
P = 3
X = 8
R = 3
num_spks = 2
gradient_checkpointing = true
```

共同训练设置：

```text
batch_size = 1（由 collate_one 和 full-utterance training 决定）
learning_rate = 1e-3
acoustic weight_decay = 1e-5
num_workers = 4
device = cuda
```

音频与 semantic feature 按 utterance ID join，不依赖目录顺序。semantic record 被整理为：

```text
semantic        [B, 2, Lmax, 2048]
semantic_mask   [B, 2, Lmax]
global_semantic [B, 2048]
```

损失里的 separation 部分采用两说话人 PIT SI-SNR：

```text
score_direct = mean(SI-SNR(output1, ref1), SI-SNR(output2, ref2))
score_swap   = mean(SI-SNR(output1, ref2), SI-SNR(output2, ref1))
L_sep        = -mean(max(score_direct, score_swap))
```

因此日志中的 `separation_loss` 越负越好，实际 dev PIT SI-SNR 等于其相反数。例如：

```text
validation/dev_separation_loss = -13.329
dev PIT SI-SNR                 =  13.329 dB
```

## 2. 实验一：Semantic Negotiation V1

### 2.1 代码与输出

```text
模型：       Conv_TasNet_Semantic.py
训练入口：   train_librimix.py
数据工具：   semantic_data.py
诊断脚本：   diagnose_pit_permutation.py
输出目录：   exp/librimix_clean_semantic
best ckpt：  exp/librimix_clean_semantic/best.pt
TensorBoard：exp/librimix_clean_semantic/tensorboard
```

### 2.2 数据路径

本地 `arguments.json` 记录：

```text
data_root:
/home/zt/Desktop/STL/espnet/egs2/librimix/sot_asr1/data

semantic_root:
/home/zt/Desktop/STL/Multi-talker-ASR-with-LLMs/semantic_features/libri2mix_clean_offset
```

### 2.3 运行命令

```bash

CUDA_VISIBLE_DEVICES=0 python train_librimix.py \
  --model semantic \
  --device cuda \
  --data-root /home/zt/Desktop/STL/espnet/egs2/librimix/sot_asr1/data \
  --semantic-root /home/zt/Desktop/STL/Multi-talker-ASR-with-LLMs/semantic_features/libri2mix_clean_offset \
  --output-dir exp/librimix_clean_semantic \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --num-workers 4
```


### 2.4 V1 模型设计

V1 先产生两路 rough acoustic sources，再计算 row-wise semantic assignment：

```text
compatibility[i,j] = cosine(Wa Pool(Ai), Ws Pool(Zj))
assignment         = softmax(compatibility, dim=semantic_stream)
```

四路 cross-attention：

```text
R11 = CrossAttn(A1, Z1)
R12 = CrossAttn(A1, Z2)
R21 = CrossAttn(A2, Z1)
R22 = CrossAttn(A2, Z2)
```

V1 同时使用 local gate、global gate 和可学习的 `alpha`：

```text
verification_gate = local_gate * global_gate
updated = acoustic + alpha * verification_gate * proposal
```

初始化：

```text
alpha = 0
local_gate bias  = -2  → sigmoid(-2) ≈ 0.119
global_gate bias = -2  → sigmoid(-2) ≈ 0.119
```

V1 只优化 PIT SI-SNR，没有显式 matching auxiliary loss；所有模型参数统一使用 `weight_decay=1e-5`。

### 2.5 V1 checkpoint 结果

`best.pt`：

```text
epoch:          3（未完成）
global_step:    130000
train loss:     -11.3863
dev loss:       -12.3177
dev PIT SI-SNR: 约 12.3177 dB
```

### 2.6 V1 观察到的现象

对完整 3000 条 dev 样本进行了 PIT permutation 诊断：

```text
PIT permutation accuracy: 50.37%
PIT target direct:         1511
PIT target swap:           1489
预测 direct:               3000
预测 swap:                 0
平均 PIT margin:           37.37 dB
```

所有样本都严格满足：

```text
P(direct) = 0.5
P(swap)   = 0.5
```

这不是有效的 50% matching，而是 direct/swap 完全平局后 `argmax` 固定返回 direct。完整逐 utterance 结果位于：

```text
exp/librimix_clean_semantic/pit_permutation_diagnostics_dev.json
```

检查 checkpoint 后发现大量 negotiation 参数范数严格为零，包括：

```text
semantic_proj
acoustic_pool_proj
semantic_pool_proj
cross_attention
global_proj
```

同时：

```text
alpha ≈ -0.00155
```

V1 退化的主要因果链推断为：

```text
alpha=0
  → 首步 semantic proposal/gate 的 separation gradient 被截断
  → alpha 后续只学到很小的数值
  → local gate × global gate 再次缩小有效梯度
  → 纯声学 separator 已能独立优化 PIT loss
  → 模型选择忽略 semantic
  → negotiation task gradient 长期很弱
  → 统一 weight decay 持续把相关参数压向 0
```

结论：V1 checkpoint 基本是纯声学 separator，semantic matching 和 cross-attention 没有形成有效作用。

## 3. 实验二：Semantic Negotiation V2

### 3.1 代码与输出

```text
模型：       Conv_TasNet_Semantic_V2.py
训练入口：   train_librimix_v2.py
输出目录：   exp/librimix_clean_semantic_v2
best ckpt：  exp/librimix_clean_semantic_v2/best.pt
TensorBoard：exp/librimix_clean_semantic_v2/tensorboard
```

### 3.3 V2 运行命令

首次初始化：只加载 V1 checkpoint 的非 `negotiation.*` 声学权重，V2 negotiation 保持全新初始化；不恢复旧 optimizer、epoch 或 global step。

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/aocheng/learning/Separation/.conda-env/bin/python \
train_librimix_v2.py \
  --device cuda \
  --data-root Separation/data/kaldi \
  --semantic-root /home/aocheng/learning/Separation/semantic/libri2mix_clean_offset \
  --output-dir Separation/exp/librimix_clean_semantic_v2 \
  --acoustic-pretrained-checkpoint /home/aocheng/learning/Separation/exp/librimix_clean_semantic/best.pt \
  --epochs 100 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --lambda-match 0.02 \
  --gradient-accumulation-steps 1 \
  --clip-norm 5.0 \
  --num-workers 4 \
  --val-interval-steps 10000
```


### 3.4 V2 相对 V1 的修改

#### 合法的 direct/swap permutation assignment

```text
score_direct = s11 + s22
score_swap   = s12 + s21
q            = softmax([score_direct, score_swap])

assignment = [[q_direct, q_swap],
              [q_swap,   q_direct]]
```

这样只允许两种一一对应排列，避免两路 acoustic outputs 同时选择同一个 semantic stream。

#### PIT permutation auxiliary loss

PIT waveform 排列被用作 matching 诊断和弱监督标签：

```text
L_match = CE([score_direct, score_swap], pit_permutation)
L_total = L_sep + 0.02 * L_match
```

#### 简化 semantic residual

V2 固定：

```text
alpha = 1
```

删除 V1 的 `global_gate/global_proj`，只保留逐 source、逐 channel、逐时间位置的 local gate：

```text
local_gate.shape = [B, 2, C, T]
updated = acoustic + local_gate * proposal
```

#### Optimizer 分组

```text
acoustic parameters:    weight_decay = 1e-5
negotiation parameters: weight_decay = 0
```

#### 声学预训练加载

从 V1 `best.pt` 加载了 302 个非 negotiation 声学张量，V2 的 17 个 negotiation state tensors 保持新初始化。

### 3.5 V2 最新 checkpoint 结果

当前 `best.pt`：

```text
epoch:                  3（未完成）
global_step:            150000
dev total loss:         -13.3154
dev separation loss:    -13.3291
dev PIT SI-SNR:          13.3291 dB
dev matching loss:        0.6849
dev matching accuracy:   58.70%
dev permutation entropy:  0.6295
dev mean PIT margin:      40.70 dB
```

训练阶段累计指标：

```text
train separation loss:  -12.3624
train matching loss:      0.6616
train matching accuracy: 60.62%
```

这里的 train metrics 是从当前 epoch 开始到验证点的累计平均；dev metrics 是每次完整 3000 条 dev 集重新评估得到的指标。

### 3.6 V2 验证趋势

```text
Step     Dev PIT SI-SNR    Dev matching accuracy
10k      12.450 dB         55.73%
50k      12.840 dB         58.43%
64.7k    12.890 dB         58.03%
90k      13.021 dB         59.17%
129.4k   13.205 dB         57.57%
150k     13.329 dB         58.70%
```

V2 的 dev SI-SNR 相对 V1 best 的约 12.318 dB 提高约 1.01 dB。但 V2 同时继续更新了 acoustic separator，因此不能把全部提升直接归因于 semantic conditioning。

### 3.7 V2 gate 现象

V2 的 TensorBoard `gate/train_mean` 是 local gate `[B,2,C,T]` 在所有维度上的均值。

```text
初始化：       sigmoid(-2) ≈ 0.119
step 1–10k：   0.0026
10k–20k：      0.0542
20k–50k：      约 0.038–0.046
50k–120k：     约 0.042–0.045
120k–130k：    0.0473
130k–150k：    约 0.053
最近 1000 步： 约 0.0527
```

因此 V2 gate 不是持续关闭，而是经历了：

```text
随机 semantic proposal 阶段快速关闭
  → matching/cross-attention 开始学习
  → gate 重新打开
  → 后期稳定在约 4%–5%
```

最近 10000 步没有任何一步的 batch gate mean 小于 0.01。

注意：gate mean 只是乘法系数，不能直接等价于 semantic 的实际贡献。更关键的指标是：

```text
||local_gate * semantic_proposal|| / ||rough_acoustic||
```

以及关闭/替换 semantic 后的 dev SI-SNR 差值。

### 3.8 V2 matching 现象

V1 的 direct/swap 概率严格为 0.5；V2 已经学到弱但可测的 matching 信号：

```text
随机 accuracy:      50%
V2 dev accuracy:    58.70%
随机二分类 CE:      log(2) ≈ 0.6931
V2 dev matching CE: 0.6849
```

这表明 matching 不再是完全随机，但约 41.3% 的 dev utterances 仍判断错误，尚不能称为可靠 assignment。

### 3.9 V2 参数与梯度现象

negotiation 参数没有像 V1 那样归零，而是持续增长。step 150k 的近似模块范数：

```text
semantic_proj:          182.67
acoustic_pool_proj:      54.14
semantic_pool_proj:      46.55
cross_attention:        110.58
local_gate.weight:       18.57
```

最新 TensorBoard 到 step 152920：

```text
semantic negotiation norm: 约 229.8
最近 1000 步 semantic grad norm: 约 0.061
最近 1000 步 matching projection grad norm: 约 0.00190
最近 1000 步 total grad norm: 约 26.16
```

全训练过程中：

```text
total grad norm > 5:  约 90.65% 的步骤
total grad norm > 10: 约 55.13% 的步骤
clip_norm:             5.0
```

因此全模型统一 gradient clipping 几乎一直生效；较大的 acoustic gradient 可能同时缩小本来较弱的 matching gradient。negotiation 使用零 weight decay 避免了归零，但参数范数长期单调增长，需要继续监控 NaN/Inf、过度自信和 dev matching loss。

## 4. 两次实验的核心对比

| 项目 | V1 | V2 |
|---|---|---|
| 声学初始化 | 从头训练 | 加载 V1 非 negotiation 权重 |
| Assignment | row-wise softmax | direct/swap permutation softmax |
| Matching supervision | 无 | PIT permutation CE，权重 0.02 |
| Alpha | 可学习，初始化 0 | 固定 1 |
| Gate | local × global | 只保留 local |
| Negotiation weight decay | 1e-5 | 0 |
| Semantic 参数状态 | 大量归零 | 未归零但范数持续增长 |
| Dev matching accuracy | 50.37%，实际为全平局 | 58.70% |
| Dev PIT SI-SNR | 约 12.318 dB | 约 13.329 dB |

## 5. 当前可以下的结论

1. V1 的 semantic branch 发生了明确塌缩，最终基本是纯声学 separator。
2. 固定 `alpha=1`、移除 global gate、对 negotiation 关闭 weight decay，并加入 PIT matching CE 后，V2 的 semantic/matching 参数可以持续学习，不再归零。
3. V2 matching 已显著脱离 V1 的严格 0.5 平局，但 58.7% accuracy 仍然偏弱。
4. V2 local gate 在训练早期先关闭，随后重新打开到约 0.05；这可能表示模型先拒绝随机 proposal，等 proposal 变得可用后再恢复 semantic 注入。
5. V2 dev SI-SNR 比 V1 高约 1 dB，但由于 acoustic 参数同时继续训练，目前不能证明这 1 dB 来自 semantic。

## 6. 当前不能直接下的结论与必要后续实验

### 6.1 Reference 与 semantic 顺序

PIT matching accuracy 的物理意义依赖：

```text
ref[0] ↔ Z1
ref[1] ↔ Z2
```

必须确认 `spk1.scp/spk2.scp` 和 SOT onset-order semantic streams 使用相同的说话人编号；否则需要先转换 reference-to-semantic permutation label。

### 6.2 Semantic ablation

应在同一个 V2 checkpoint、同一 dev 集上评估：

```text
1. 正常 semantic
2. semantic 全零
3. Z1/Z2 交换
4. 其他 utterance semantic replacement
5. 强制 semantic residual = 0（updated = acoustic）
```

如果关闭或替换 semantic 后 SI-SNR 明显下降，才能证明 semantic conditioning 对 separation 有实际贡献。

### 6.3 独立 test 集

`validation/dev_separation_loss` 是完整 dev 集 PIT SI-SNR 的负值，但 dev 已被用于选择 `best.pt` 和调参，不能替代最终 test。模型和超参数确定后，应只在独立 test set 上进行最终一次评估。

### 6.4 推荐继续记录

```text
matching accuracy / CE / entropy
local gate mean 与分位数
||gate * proposal|| / ||rough_acoustic||
各 negotiation 模块 parameter norm / gradient norm
正常、zero、swap、replacement semantic 下的 SI-SNR 差值
```
