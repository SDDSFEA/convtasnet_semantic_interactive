# Conv-TasNet Semantic–Acoustic Negotiation 改版说明

## 目标

在原始 Conv-TasNet 上加入缓存的 SOT LLM-ASR decoder hidden states，但不把 ASR
语义当成可靠 ground truth。模型先生成粗分离声源，再完成 source–semantic matching，
最后只接收被声学状态支持的语义 proposal。

本改版保留 `Conv_TasNet.py` 不变，新增：

- `Conv_TasNet_Semantic.py`：模型及 negotiation block；
- `semantic_data.py`：按 utterance ID 读取、padding 两路语义 token；
- `options/train/train_semantic_example.yml`：建议配置。

## 网络流程

```text
mixture [B,T]
  -> encoder [B,N,Te]
  -> bottleneck [B,C,Te]
  -> front TCN
  -> rough masks [B,2,N,Te]
  -> rough acoustic sources A1,A2 [B,2,C,Te]
  -> 2x2 source-semantic matching
  -> token-to-frame cross-attention proposals
  -> local/global acoustic verification gates
  -> source-specific back TCN
  -> final masks [B,2,N,Te]
  -> decoder -> waveform 1, waveform 2
```

默认 `R=3, front_repeats=1`：第一个 repeat 只依赖声学信息，后两个 repeats
用于 negotiation 后的 source-specific refinement。

## 为什么不直接复制 IFF-Net

IFF-Net 的 enhanced/noisy Fbank 都是 `[B,C,T,F]`，可以直接 concat、逐点相乘和
conv gate。这里的输入是：

```text
acoustic Ai: [B,C,Te]
semantic Zj: [B,Lj,2048]
```

`Te` 是声学帧，`Lj` 是非对齐 token；`<sc>` 也不是声学时间边界。因此禁止将
semantic token 线性插值到声学时间轴。先通过 cross-attention 把 `Zj` 转换为
`[B,Te,C]` 的 acoustic-domain proposal `Rij`，之后才能使用 IFF 风格的 gate。

## 2x2 Soft Matching

Conv-TasNet 输出 permutation 与 SOT 顺序不保证一致，因此不固定
`A1<->Z1, A2<->Z2`：

```text
sij = cosine(Wa Pool(Ai), Ws MaskedMean(Zj))
P   = Softmax(S, dim=semantic_stream)
```

`P` 的 shape 是 `[B,2,2]`。第一版使用 row-wise softmax；Sinkhorn 作为后续
ablation，避免第一版引入过多变量。

## Semantic Proposal 与 Verification

```text
Rij = CrossAttn(query=Ai, key=Zj, value=Zj)
Ri  = sum_j Pij Rij

Gi_local = sigmoid(Conv1x1[Ai, Ri, Ai*Ri, |Ai-Ri|])
gi_global = sigmoid(MLP[Pool(Ai), Pool(Ri), GlobalSemantic])
Gi = Gi_local * gi_global

Ai' = Ai + alpha * Gi * Ri
```

local/global gate 最后一层 bias 初始化为 `-2`，`alpha` 初始化为 0，使模型起始
状态接近纯声学路径，降低错误 ASR semantic 在训练早期破坏 separator 的风险。

## 已缓存语义特征的使用

每条 `.pt` 包含：

```text
speaker1_hidden [L1,2048]
speaker2_hidden [L2,2048]
global_pooled   [2048]
sc_count
```

`SemanticFeatureStore` 从 split 或 train shard manifests 建立 ID 索引；
`collate_semantic_records` padding 成：

```text
semantic        [B,2,Lmax,2048]
semantic_mask   [B,2,Lmax]
global_semantic [B,2048]
```

音频 dataset 必须保留 mixture ID，并与 semantic store 用同一 ID join。不要依赖
目录顺序配对。

## 重要的数据切块限制

现有 baseline `DataLoaders.py` 把音频随机切成 4 秒，但缓存语义描述完整 utterance。
如果把完整语义直接提供给任意 4 秒 chunk，会让模型看到 chunk 外的文本内容，产生
训练/评估不一致和信息泄漏。

第一版建议使用 full-utterance batching + audio padding/mask。若必须 chunk training，
需要以下方案之一：

1. 重新提取 chunk-level ASR semantic；或
2. 通过可靠 token timestamp 只选择落在 chunk 内的 token；或
3. 明确把 full-context conditioning 作为任务设定，并保证 train/dev/test 完全一致。

推荐方案 1，最容易保证实验结论干净。

## 训练阶段

### Stage 0：纯声学初始化

从 baseline Conv-TasNet checkpoint 初始化 encoder、bottleneck、front/back TCN 和
decoder。rough/final heads 与 negotiation 新增参数单独初始化。由于 baseline 只有
共享 TCN，而改版后 back TCN 对两个 source 共享执行，需要按模块名映射权重，不能
直接 `strict=True` 加载整个 checkpoint。

### Stage 1：最小可行模型

- 冻结缓存 ASR 特征；
- 只用 waveform PIT SI-SNR/SI-SDR loss；
- 训练 matching、cross-attention、gate 与 separator；
- 不做 reverse feedback；
- 记录 assignment entropy、gate mean、gate histogram。

### Stage 2：错误语义拒绝训练

按一定概率执行：

- swap `Z1/Z2`（预期 assignment 跟随交换，不应作为“错误拒绝”）；
- drop token；
- 用其他 utterance 的 stream replacement；
- hidden-state noise；
- whole-semantic dropout。

真正验证 gate 的是 replacement/hallucination/drop，而不是 swap；swap 主要验证
soft matching 是否解决 permutation。

### Stage 3：可选反向交互

只有 Stage 1/2 明确有效后，加入 acoustic-to-semantic one-step update。缓存的原始
semantic 不修改，反向更新只存在于当前 forward 内。

## 损失与诊断

第一版优先：

```text
L = L_PIT-SI-SNR
```

不要一开始强制 `P` 对齐 SOT/source index，因为 separator permutation 本来就是
任意的。可在 PIT 找到 waveform permutation 后，再把它作为弱 matching target：

```text
L = L_sep + lambda_match * CE(P, pit_permutation)
```

建议保存：

- `assignment [B,2,2]`；
- `verification_gate [B,2,C,Te]` 的均值/分位数；
- semantic corruption type；
- clean/corrupted semantic 下 SI-SDR 差值；
- rough mask 与 final mask 的 SI-SDR。

## 必做 Ablation

1. 原始 Conv-TasNet；
2. direct global semantic conditioning；
3. cross-attention，但无 matching/gate；
4. + 2x2 soft matching；
5. + verification gate（完整第一版）；
6. + semantic corruption training；
7. oracle transcript hidden 与 predicted transcript hidden；
8. speaker streams 与只用 `global_pooled`。

## 当前实现边界

- 当前只支持 2 speakers；
- 没有 reverse feedback；
- 没有把 semantic loader 接进旧的随机 chunk `DataLoaders.py`，这是有意避免
  full-utterance semantic 泄漏到局部 chunk；
- 模型 forward 和 backward smoke test 已覆盖 variable token mask、2x2 matching、
  cross-attention、gate、rough/final masks 和 waveform reconstruction。

## 可运行训练命令

统一入口 `train_librimix.py` 使用之前准备的 ESPnet offset Libri2Mix clean
`train/dev`，采用 full utterance、batch size 1，并通过 utterance ID join semantic。

Baseline：

```bash
cd /home/zt/Desktop/STL/Conv-TasNet/Conv_TasNet_Pytorch
CUDA_VISIBLE_DEVICES=0 python train_librimix.py \
  --model baseline \
  --output-dir exp/librimix_clean_baseline \
  --epochs 100 \
  --num-workers 4
```

Semantic negotiation：

```bash
cd /home/zt/Desktop/STL/Conv-TasNet/Conv_TasNet_Pytorch
CUDA_VISIBLE_DEVICES=0 python train_librimix.py \
  --model semantic \
  --output-dir exp/librimix_clean_semantic \
  --epochs 100 \
  --num-workers 4
```

断点恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python train_librimix.py \
  --model semantic \
  --output-dir exp/librimix_clean_semantic \
  --epochs 100 \
  --num-workers 4 \
  --resume exp/librimix_clean_semantic/last.pt
```

TensorBoard：

```bash
tensorboard --logdir /home/zt/Desktop/STL/Conv-TasNet/Conv_TasNet_Pytorch/exp \
  --port 6006
```

每个实验的 `tensorboard/` 目录记录：

- `loss/train_step`；
- `loss/dev_step`；
- `loss/epoch` 中的 train/dev 均值；
- `optimizer/learning_rate`。
