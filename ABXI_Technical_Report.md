# ABXI 模型技术深度分析报告

## 1. 模型概述

**ABXI** (Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation) 是 WWW 2025 发表的跨域序列推荐模型。其核心创新在于通过**双路径LoRA适配机制**实现域特定适应与任务引导不变兴趣学习的解耦。

---

## 2. 核心架构设计

### 2.1 整体架构

ABXI采用**单编码器+双适配器**的简洁架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     输入层                                   │
│  seq_x (混合序列)  seq_a (域A序列)  seq_b (域B序列)         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Embedding Layer (共享)                     │
│              Item Embedding + Position Embedding            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Shared Multi-Head Attention (共享)              │
│                  捕获通用序列模式                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│   FFN + dlora_x │  │  FFN + dlora_a │  │  FFN + dlora_b │
│   (混合序列)    │  │   (域A序列)     │  │   (域B序列)    │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        ↓                   ↓                   ↓
┌─────────────────────────────────────────────────────────────┐
│              Projector + Invariant LoRA (ilora)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  proj_i     │  │  ilora_a    │  │  ilora_b    │          │
│  │ (不变兴趣)   │  │ (域A适配)   │  │ (域B适配)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
           域A输出      混合表示       域B输出
```

### 2.2 关键组件详解

#### 2.2.1 Domain LoRA (dlora) - 域特定适配

**结构定义** (layers.py):
```python
class LoRA(nn.Module):
    def __init__(self, d_embed, rank=16):
        # 低秩分解: W + BA
        self.mat_A = nn.Parameter(torch.randn(d_embed, rank) / 50)
        self.mat_B = nn.Parameter(torch.zeros(rank, d_embed))
```

**三个独立LoRA**：
- `dlora_x`: 适配混合序列
- `dlora_a`: 适配域A序列
- `dlora_b`: 适配域B序列

**初始化策略**：
- `mat_A`: 随机初始化，缩放1/50
- `mat_B`: **零初始化** → 训练开始时LoRA不改变原网络行为

#### 2.2.2 Invariant LoRA (ilora) - 任务引导适配

**核心创新**：根据ground truth所属域动态激活

```python
# ABXI.py:85-96
h_i = self.proj_i(h_x)  # 投影到不变兴趣空间

# 域A: 混合序列 + 不变兴趣 + 域A特定适配
h_a = (self.norm_i2a((h_x + self.dropout(h_i) + self.dropout(self.ilora_a(h_x))) * mask_gt_a) +
       self.norm_a2a((h_a + self.dropout(self.proj_a(h_a))) * mask_gt_a))
```

**设计解读**：
- `h_i = proj_i(h_x)`: 从混合序列提取**域不变兴趣**
- `ilora_a(h_x)`: 为域A预测任务学习特定的低秩适配
- `mask_gt_a`: **只在ground truth属于域A时生效** → 任务引导

#### 2.2.3 投影器设计

```python
# ABXI.py:35-44
self.proj_i = LoRA(d_embed, rank=ri)      # 不变兴趣投影
self.proj_a = nn.Linear(d_embed, d_embed)  # 域A特定投影
self.proj_b = nn.Linear(d_embed, d_embed)  # 域B特定投影
```

**关键区别**：
- `proj_i` 使用LoRA：学习低秩的不变兴趣表示
- `proj_a/b` 使用标准Linear：充分表达域特定特征

### 2.3 前向传播详解

```python
def forward(self, seq_x, seq_a, seq_b, mask_x, mask_a, mask_b, mask_gt_a, mask_gt_b):
    # 1. 嵌入层
    h_x = self.dropout((self.ei(seq_x) + self.embed_pos(mask_x)) * mask_x)
    h_a = self.dropout((self.ei(seq_a) + self.embed_pos(mask_a)) * mask_a)
    h_b = self.dropout((self.ei(seq_b) + self.embed_pos(mask_b)) * mask_b)

    # 2. 共享的Multi-Head Attention
    h_x = self.mha(h_x, mask_x)
    h_a = self.mha(h_a, mask_a)
    h_b = self.mha(h_b, mask_b)

    # 3. FFN + Domain LoRA
    h_x = self.norm_sa_x(h_x + self.dropout(self.ffn(h_x)) + self.dropout(self.dlora_x(h_x)))
    h_a = self.norm_sa_a(h_a + self.dropout(self.ffn(h_a)) + self.dropout(self.dlora_a(h_a)))
    h_b = self.norm_sa_b(h_b + self.dropout(self.ffn(h_b)) + self.dropout(self.dlora_b(h_b)))

    # 4. Projector + Invariant LoRA (核心创新)
    h_i = self.proj_i(h_x)

    # 域A表示 = 不变兴趣通路 + 域特定通路
    h_a = (self.norm_i2a((h_x + self.dropout(h_i) + self.dropout(self.ilora_a(h_x))) * mask_gt_a) +
           self.norm_a2a((h_a + self.dropout(self.proj_a(h_a))) * mask_gt_a))

    # 域B表示 = 不变兴趣通路 + 域特定通路
    h_b = (self.norm_i2b((h_x + self.dropout(h_i) + self.dropout(self.ilora_b(h_x))) * mask_gt_b) +
           self.norm_b2b((h_b + self.dropout(self.proj_b(h_b))) * mask_gt_b))

    # 5. 根据任务选择输出
    h = h_a * mask_gt_a + h_b * mask_gt_b
    return h
```

---

## 3. 域内与跨域信息交换机制

### 3.1 信息交换架构图

```
┌────────────────────────────────────────────────────────────┐
│                      共享编码层                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Shared MHA + Shared FFN                            │  │
│  │   (所有序列共享相同参数)                              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  dlora_x │     │  dlora_a │     │  dlora_b │  ← Domain LoRA
   └────┬─────┘     └────┬─────┘     └────┬─────┘
        │                │                │
        │    ┌───────────┴───────────┐    │
        │    ↓                       ↓    │
        │  proj_i                 proj_a/b │
        │    │                       │    │
        │    ↓                       ↓    │
        │  ilora_a/b               (输出) │
        │    │                            │
        └────┼────────────────────────────┘
             ↓
      ┌──────────────┐
      │  最终表示 h   │ = 不变兴趣 + 域特定
      └──────────────┘
```

### 3.2 双路径信息流动

ABXI的信息交换通过**两条路径**实现：

| 路径 | 名称 | 机制 | 功能 |
|------|------|------|------|
| **Path-I** | Invariant Path | `h_x → proj_i → h_i` | 提取跨域共享的不变兴趣 |
| **Path-S** | Specific Path | `h_a → proj_a → h_a` | 保留域特定的特征 |

**融合策略**：
```python
h_a_final = normalize( h_x + h_i + ilora_a(h_x) )  # 不变通路
          + normalize( h_a + proj_a(h_a) )          # 特定通路
```

### 3.3 任务引导机制

**核心设计**：`mask_gt_a` 和 `mask_gt_b` 的使用

```python
# 只在ground truth属于对应域时才计算该域的适配
h_a = (... * mask_gt_a) + (... * mask_gt_a)
```

**效果**：
- 当预测目标属于域A时，`mask_gt_a=1`, `mask_gt_b=0`
- 模型只优化域A的表示，域B分支被屏蔽
- 实现**任务感知的动态适配**

---

## 4. Loss 设计分析

### 4.1 简单的联合损失

```python
def train_batch(self, batch):
    seq_x, seq_a, seq_b, gt, gt_neg = batch

    # 前向传播
    h = self.model(seq_x, seq_a, seq_b, mask_x, mask_a, mask_b, mask_gt_a, mask_gt_b)

    # 计算损失
    loss_a, loss_b = self.model.cal_rec_loss(h, gt, gt_neg, mask_gt_a, mask_gt_b)
    (loss_a + loss_b).backward()
```

### 4.2 分域损失计算

```python
def cal_rec_loss(self, h, gt, gt_neg, mask_gt_a, mask_gt_b):
    # 域A和域B的mask
    # 分别计算两个域的InfoNCE损失

    loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
    loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()
```

**与MERIT的区别**：
- ABXI只有**单一输出** `h`，没有多分支监督
- 依赖LoRA的适配能力来学习跨域知识

### 4.3 温度系数

```python
temp = 0.75  # 与MERIT相同
```

---

## 5. 与MERIT的关键差异

### 5.1 架构复杂度对比

| 维度 | ABXI | MERIT |
|------|------|-------|
| 编码器 | 1个共享 | 3个独立 |
| 注意力层 | 单层 | 单层+交叉注意力 |
| 适配机制 | LoRA (低秩) | MoFFN (门控专家) |
| 信息融合 | 投影器相加 | 交叉注意力 |
| 表示数量 | 1个最终表示 | 3个表示 (h_m, h_a, h_b) |

### 5.2 参数效率

ABXI的**LoRA设计**更加参数高效：

```python
# LoRA参数量
# 每个LoRA: d_embed * rank * 2
# 共6个LoRA (3 dlora + 2 ilora + 1 proj_i)
total_lora_params = 6 * d_embed * rank * 2

# 当 d_embed=256, rank=8 时
# total_lora_params = 6 * 256 * 8 * 2 = 24,576
```

相比MERIT的完整MoFFN（4个专家网络），ABXI的参数开销更小。

---

## 6. 超参数与工程细节

### 6.1 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_embed` | 256 | 嵌入维度 |
| `n_attn` | 1 | Transformer层数 |
| `n_head` | 2 | 注意力头数 |
| `dropout` | 0.5 | Dropout率 |
| `rd` | 8 | Domain LoRA秩 |
| `ri` | 8 | Invariant LoRA秩 |
| `temp` | 0.75 | 温度系数 |

### 6.2 低秩秩的选择

```python
parser.add_argument('--rd', type=int, default=8)  # 域LoRA秩
parser.add_argument('--ri', type=int, default=8)  # 不变LoRA秩
```

**设计选择**：
- 较小的秩(8)保持参数效率
- 域LoRA和不变LoRA使用相同秩，平衡两种适配能力

### 6.3 优化策略

```python
optimizer = AdamW(lr=1e-4, weight_decay=5e0)
scheduler_warmup = LinearLR(start_factor=1e-5, end_factor=1., total_iters=n_warmup)
scheduler = ReduceLROnPlateau(mode='max', factor=0.3162, patience=30)
```

**注意**：`lr_g=0.3162` ≈ `10^(-0.5)`，即学习率每次衰减为原来的1/√10

---

## 7. 核心设计亮点总结

1. **双LoRA机制**：
   - Domain LoRA：学习域特定的低秩适配
   - Invariant LoRA：学习任务引导的不变兴趣适配

2. **任务引导的掩码机制**：
   - `mask_gt_a/b` 实现任务感知的动态表示选择
   - 避免无关域的梯度干扰

3. **参数效率**：
   - 低秩适配大幅减少了跨域学习的参数量
   - 适合数据稀疏场景

4. **简洁的架构设计**：
   - 单编码器 + 双适配器
   - 易于实现和调优

