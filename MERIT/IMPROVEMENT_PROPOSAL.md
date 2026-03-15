# MERIT 模型改进方案：渐进式跨域信息压缩架构

## 一、当前MERIT架构的局限

### 1.1 硬分割问题
当前MERIT通过ID阈值将序列硬切分为三个独立流：
```python
mask_a = (seq_m > self.n_item_a)      # Domain B
mask_b = (seq_m <= self.n_item_a)     # Domain A
```
- **问题1**：三个独立的Self-Attention层，参数量冗余（3倍）
- **问题2**：Domain A和B在SA层完全隔离，无法捕获早期跨域关联
- **问题3**：域边界是人工预定义的，缺乏灵活性

### 1.2 晚融合问题
MERIT采用"先隔离后融合"策略：
- 信息隔离：三个流独立处理多层
- 一次性融合：只在最后的Cross-Attention层交互
- 丢失细粒度跨域协同信号

---

## 二、改进方案概述

### 2.1 核心设计思想

采用**渐进式跨域信息压缩**（Progressive Cross-Domain Information Compression）策略：
1. **统一架构**：单Transformer处理所有域，消除硬边界
2. **RoPE域偏置**：用可学习的注意力偏置替代硬mask
3. **目标域条件**：注入目标域编号进行条件预测
4. **分层压缩**：从远到近逐层压缩跨域信息

### 2.2 架构对比

```
MERIT (原始):
┌─────────┐    ┌─────────┐    ┌─────────┐
│  SA_A   │    │  SA_B   │    │  SA_M   │  ← 三个独立SA层
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     └──────────────┼──────────────┘
                    ▼
              Cross-Attention  ← 一次性融合

改进方案:
┌─────────────────────────────────────────────┐
│  Unified Transformer + Progressive Mask     │
│                                             │
│  Layer 0: Full visibility                   │
│     └── 看到全部跨域历史                    │
│                                             │
│  Layer 1: Compressed visibility             │
│     └── 看到 [倒数第2个同域] ~ [最后] 之间 │
│                                             │
│  Layer 2: Local visibility                  │
│     └── 看到 [倒数第1个同域] ~ [最后] 之间 │
└─────────────────────────────────────────────┘
```

---

## 三、详细设计方案

### 3.1 RoPE域感知注意力偏置

用可学习的注意力偏置替代硬mask，允许模型动态学习跨域关联强度：

```python
class DomainAwareAttention(nn.Module):
    def __init__(self, d_embed, n_heads):
        self.rope = RotaryPositionalEmbedding(d_embed)
        # 可学习的域间注意力偏置
        self.domain_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))

    def forward(self, q, k, v, domain_ids):
        # RoPE位置编码
        q = self.rope(q)
        k = self.rope(k)

        # 计算注意力分数
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / sqrt(dk)

        # 添加域偏置：同域=0，跨域=learnable_bias
        domain_mask = (domain_ids.unsqueeze(-1) != domain_ids.unsqueeze(1))
        domain_mask = domain_mask.unsqueeze(1)  # [B, 1, seq, seq]
        attn_logits = attn_logits + self.domain_bias * domain_mask.float()

        attn_weights = F.softmax(attn_logits, dim=-1)
        return attn_weights @ v
```

**关键优势**：
- 软边界：跨域物品仍可建立注意力连接（强度由模型学习）
- 可解释性：domain_bias的数值反映模型对跨域关联的置信度
- 渐进启用：初期bias接近0，随训练逐渐学习

### 3.2 目标域条件注入

在输入中明确告知模型要预测的目标域，实现条件序列建模：

```python
class ConditionalDomainEmbedding(nn.Module):
    def __init__(self, n_items, n_domains, d_embed, n_prompt_tokens=4):
        self.item_emb = nn.Embedding(n_items, d_embed)
        self.pos_emb = nn.Embedding(max_seq_len, d_embed)

        # 为每个域学习一组prompt token
        self.domain_prompts = nn.Parameter(
            torch.randn(n_domains, n_prompt_tokens, d_embed)
        )

    def forward(self, seq_m, target_domain):
        """
        seq_m: [batch, seq_len] item IDs
        target_domain: [batch] 目标域编号 (0 or 1)
        """
        batch_size = seq_m.size(0)

        # 获取目标域的prompt
        prompts = self.domain_prompts[target_domain]  # [B, n_prompt, d_embed]

        # Item embedding
        items = self.item_emb(seq_m)
        positions = self.pos_emb(torch.arange(seq_m.size(1), device=seq_m.device))

        # 拼接：[prompts | items]
        h = torch.cat([prompts, items + positions], dim=1)

        return h, n_prompt_tokens  # 返回prompt长度用于后续剥离
```

**收益**：
- 明确条件：模型知道要预测Domain A还是B，输出分布更聚焦
- 支持迁移：训练时见过的域组合可以zero-shot迁移
- 降低混淆：避免梯度冲突，不同域的预测目标不互相干扰

### 3.3 渐进式跨域信息压缩（核心创新）

#### 3.3.1 核心思想

从远到近逐层压缩跨域信息，每层将远程跨域行为"聚合"到最近的同域锚点：

```
示例序列：[B1, A1, B2, A2, B3, A3] （A是目标域，要预测A4）

Layer 0 (Full Visibility):
  A3可以看到: [B1, A1, B2, A2, B3] 全部历史
  └── 捕获长程跨域依赖（如：B1的长期影响）

Layer 1 (Compressed):
  A3可以看到: [A2, B3] 之间
  └── A2已经聚合了[B1, B2]的信息，A3只关注近期

Layer 2 (Local):
  A3可以看到: [B3] 紧邻
  └── 最直接的时间邻近影响
```

#### 3.3.2 Mask生成算法

```python
def generate_progressive_masks(seq_len, n_layers, domain_ids, target_domain):
    """
    生成渐进式压缩mask

    Args:
        seq_len: 序列长度
        n_layers: Transformer层数
        domain_ids: [seq_len] 每个位置的域编号
        target_domain: 目标域编号（要预测的域）

    Returns:
        masks: [n_layers, seq_len, seq_len] 每层的注意力mask
    """
    masks = []

    # 找到目标域的所有位置（作为"锚点"）
    anchor_positions = [i for i, d in enumerate(domain_ids) if d == target_domain]

    for layer_idx in range(n_layers):
        mask = torch.zeros(seq_len, seq_len)
        compression_ratio = 2 ** layer_idx  # 每层压缩率翻倍

        for i, anchor_pos in enumerate(anchor_positions):
            if i == 0:
                # 第一个锚点可以看到之前全部
                visible_start = 0
            else:
                # 后续锚点根据层数决定视野范围
                prev_anchor = anchor_positions[i - 1]
                window_size = (anchor_pos - prev_anchor) // compression_ratio
                visible_start = max(0, anchor_pos - window_size)

            # 锚点可以看到 [visible_start, anchor_pos) 的所有位置
            mask[anchor_pos, visible_start:anchor_pos] = 1

        masks.append(mask)

    return torch.stack(masks)  # [n_layers, seq_len, seq_len]
```

#### 3.3.3 物理意义

| 层级 | 视野范围 | 建模的依赖类型 | 类比 |
|------|----------|----------------|------|
| Layer 0 | Full | 长期跨域兴趣画像 | 长期记忆 |
| Layer 1 | 1/2 | 阶段性跨域偏好 | 工作记忆 |
| Layer 2 | 1/4 | 即时跨域上下文 | 短期注意 |
| ... | ... | ... | ... |

### 3.4 完整模型架构

```python
class ImprovedCrossDomainModel(nn.Module):
    def __init__(self, n_items_a, n_items_b, d_embed=256, n_layers=4, n_heads=8):
        super().__init__()
        self.n_items = n_items_a + n_items_b
        self.n_domains = 2
        self.d_embed = d_embed
        self.n_layers = n_layers

        # 统一embedding层
        self.domain_cond_embed = ConditionalDomainEmbedding(
            self.n_items, self.n_domains, d_embed
        )

        # Transformer层：统一用DomainAwareAttention
        self.layers = nn.ModuleList([
            TransformerLayer(d_embed, n_heads, domain_aware=True)
            for _ in range(n_layers)
        ])

        # 输出层
        self.output_proj = nn.Linear(d_embed, self.n_items)

    def forward(self, seq_m, domain_ids, target_domain):
        # 1. 条件嵌入 [prompts | items]
        h, n_prompt = self.domain_cond_embed(seq_m, target_domain)
        seq_len = h.size(1)

        # 2. 生成渐进式mask [n_layers, seq_len, seq_len]
        progressive_masks = generate_progressive_masks(
            seq_len, self.n_layers, domain_ids, target_domain
        )

        # 3. 逐层处理（每层用不同的mask）
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, domain_ids, progressive_masks[layer_idx])

        # 4. 剥离prompt，只取item位置的输出
        h = h[:, n_prompt:]  # [B, seq_len, d_embed]

        # 5. 预测（只取最后一个位置用于next item prediction）
        logits = self.output_proj(h[:, -1, :])  # [B, n_items]

        return logits
```

---

## 四、理论收益分析

### 4.1 多尺度时序建模

**传统方法**：所有层用同样的attention window，无法区分长期/短期依赖。

**改进方案**：每层有不同的"有效感受野"：
- Layer 0（宽）：捕捉"半年前在B域的偏好如何影响现在A域选择"
- Layer 1（中）：捕捉"近期B域行为对A域的影响"
- Layer 2（窄）：捕捉"上一个B域行为对下一个A域行为的直接影响"

### 4.2 信息瓶颈与层次压缩

每一层都在做有损压缩，将跨域历史聚合到同域锚点：

```
Layer 0 → Layer 1: 把 [B1, B2, ..., Bn] 的信息压缩进 A2
Layer 1 → Layer 2: 把压缩后的信息 + [B3] 进一步压缩进 A3

最终：所有跨域历史被压缩成紧邻的"上下文向量"
```

**理论依据**：Information Bottleneck Principle（Tishby et al.），最优表示应在压缩和预测力之间平衡。

### 4.3 因果一致性

```
标准Cross-Attention：可能破坏因果（如果B在A之后）
渐进压缩：         压缩机制天然防止未来信息泄露

假设序列：[A1, B1, A2, ?]
Layer 0: A2作为query，只能attend [A1, B1]（合法）
Layer 1: ?作为query，attend压缩后的表示（不含未来）
```

### 4.4 计算效率

```
标准Full Attention: O(L²) per layer，总复杂度 O(n_layers × L²)
渐进压缩:
  Layer 0: O(L × L) = O(L²)
  Layer 1: O(L/2 × L/2) ≈ O(L²/4)
  Layer 2: O(L/4 × L/4) ≈ O(L²/16)
  ...

总复杂度 ≈ O(L²) × (1 + 1/4 + 1/16 + ...) ≈ O(1.33L²)
```

虽然理论同阶，但实际稀疏度随层增加，可配合Sparse Attention进一步加速。

---

## 五、实验设计建议

### 5.1 消融实验

| 实验组 | 配置 | 目的 |
|--------|------|------|
| Baseline | 原始MERIT | 验证改进必要性 |
| Variant A | 仅RoPE偏置（无渐进压缩） | 验证软边界收益 |
| Variant B | 仅目标域条件 | 验证条件建模收益 |
| Variant C | RoPE + 条件（无渐进压缩） | 验证前两者的组合 |
| **Full Model** | **RoPE + 条件 + 渐进压缩** | **完整方案** |

### 5.2 超参分析

**渐进压缩率**：
- 固定压缩率（每层/2）vs 自适应压缩率
- 验证最优压缩曲线

**目标域Prompt长度**：
- n_prompt_tokens ∈ {1, 2, 4, 8}

**域偏置初始化**：
- 初始bias=0（软启动）vs 初始bias=-inf（硬mask初始）

### 5.3 可视化分析

1. **注意力热力图**：观察每层实际学到的注意力模式
2. **Domain Bias演化**：训练过程中bias数值的变化
3. **跨域迁移分析**：在训练时没见过的域组合上的zero-shot表现

---

## 六、潜在风险与对策

### 6.1 信息丢失风险

**问题**：早期层压缩可能丢失关键信息。

**对策**：
```python
# 引入注意力权重保留重要token
compression_weight = softmax(attention_scores_to_cross_domain)
compressed_repr = sum(w_i * B_i * importance_weight[i])
```

### 6.2 动态路由开销

**问题**：每层重新计算mask可能有开销。

**对策**：
```python
# 预计算mask模式，training/inference时lookup
self.register_buffer('layer_masks', precomputed_masks)
```

### 6.3 序列长度不匹配

**问题**：序列长度不能被2^layer整除。

**对策**：
```python
# 基于实际anchor位置动态计算，而非固定比例
visible_start = prev_anchor_pos if exists else 0
window_size = (anchor_pos - prev_anchor) // compression_ratio
```

---

## 七、与相关工作的联系

### 7.1 OneTrans（京东，2024）
- **OneTrans核心**：统一处理S-token（序列）和NS-token（非序列）
- **本方案联系**：将Domain A/B看作异构token，统一架构处理
- **关键差异**：OneTrans是空间上的统一，本方案是时间上的分层压缩

### 7.2 LLM领域借鉴
- **RoPE**：LLaMA/PaLM的位置编码
- **Prompt Tuning**：为每个域学习连续向量
- **Mixture of Experts**：软路由替代硬分割

### 7.3 认知科学基础
- **工作记忆理论**：人类的多尺度记忆机制
- **注意力分配**：从宽到窄的注意聚焦过程

---

## 八、总结

本改进方案通过以下三点提升MERIT：

1. **RoPE域偏置**：打破硬边界，允许动态跨域关联
2. **目标域条件**：明确预测目标，降低多任务混淆
3. **渐进式压缩**：从远到近分层聚合跨域信息，符合多尺度时序建模原理

**核心洞察**：从"先隔离后融合"转向"渐进式条件融合"，更符合人类认知顺序：先理解域内偏好，再考虑跨域影响，且明确知道当前要预测的目标域。

**预期收益**：
- 更好的跨域迁移能力
- 更细粒度的时序依赖建模
- 更高的计算效率
- 更强的可解释性
