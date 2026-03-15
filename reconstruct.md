# AutoCDSR 模型启动命令映射

## 启动命令
```bash
python src/train.py trainer=ddp experiment=kuairand model=hf_transformer_cd_sid_ib_kuairand_pareto
```

## 模型配置链

### 1. 主配置文件
**文件路径:** `AutoCDSR/configs/model/hf_transformer_cd_sid_ib_kuairand_pareto.yaml`

该文件继承自 `hf_transformer_cd_sid_ib_kuairand`，并添加了Pareto多任务优化插件。

### 2. 基础配置文件
**文件路径:** `AutoCDSR/configs/model/hf_transformer_cd_sid_ib_kuairand.yaml`

内容：
```yaml
_target_: src.models.modules.hf_transformer_module_cross_domain_sparse_id.HFCDSIDIBTransformerModule
```

## 模型代码文件

**文件路径:** `AutoCDSR/src/models/modules/hf_transformer_module_cross_domain_sparse_id.py`

**模型类名:** `HFCDSIDIBTransformerModule`

### 相关文件
- `AutoCDSR/src/models/modules/hf_transformer_base_module.py` - 基础Transformer模块
- `AutoCDSR/src/models/modules/hf_transformer_module.py` - 标准Transformer模块
- `AutoCDSR/src/models/models/modeling_t5.py` - T5编码器模型实现

## 组件说明
- `cd_sid` = Cross-Domain Sparse ID（跨域稀疏ID）
- `ib` = Information Bottleneck（信息瓶颈）
- `pareto` = Pareto优化多任务学习

## 项目信息
- **论文:** Revisiting Self-attention for Cross-domain Sequential Recommendation
- **会议:** KDD 2025

---

# 代码问题分析：域区分机制

## 问题
该模型如何区分数据来自哪个域？

## 核心机制解析

### 1. 输入层：domain_ids 参数
**代码位置:** `forward()` 方法 (第348-354行)

```python
def forward(
    self,
    input_ids: torch.Tensor,
    domain_ids: torch.Tensor,  # ← 域标识输入
    attention_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
```

**分析:** 除了常规的 `input_ids` 和 `attention_mask`，模型接收 `domain_ids` 作为显式的域标识。这是一个与 `input_ids` 同形状的tensor，每个位置标注了对应token所属的域ID。

---

### 2. 掩码生成：create_domain_mask()
**代码位置:** 第185-207行

```python
def create_domain_mask(
    self,
    input_ids: torch.Tensor,
    domain_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    domain_masks = {}
    for domain in self.domain_encoders:  # 遍历所有域
        # 条件：是mask token 或 属于当前domain
        domain_mask = torch.logical_or(
            input_ids == self.masking_token,  # 被mask的token
            domain_ids == int(domain),        # 当前域的token
        )
        domain_masks[domain] = domain_mask.to(self.embedding_table.weight.device)
    return domain_masks
```

**分析:**
- 为每个域生成一个布尔掩码 (True表示该token属于此域或是mask token)
- **关键设计:** Mask token被包含在所有域中（因为它可能来自任何域）
- 返回Dict，键是域名，值是对应的掩码tensor

---

### 3. 域特定编码器：domain_encoders
**代码位置:** `__init__` 第44-51行

```python
self.domain_encoders = torch.nn.ModuleDict({
    domain: self._spawn_domain_encoder_as_module_list(
        domain_encoder=domain_models[domain],
    )
    for domain in domain_models
})
```

**分析:**
- 使用 `torch.nn.ModuleDict` 存储每个域的独立编码器
- **键是字符串化的域ID**，值是该域的Transformer层序列
- 每个域拥有完全独立的参数

---

### 4. IB (Information Bottleneck) 跨域通信
**代码位置:** `cross_domain_ib_forward()` 第268-346行

**核心逻辑:**
1. **IB Token初始化** (第286-296行): 从embedding table中取出预定义的IB token (token id 2开始，因为0/1是padding/masking)
   ```python
   ib_embedding = self.embedding_table(
       torch.arange(2, 2 + self.num_ib_tokens).to(...)
   )
   embeddings = torch.cat([ib_embedding, inputs_embeds], dim=1)
   ```

2. **域掩码扩展** (第301-309行): 让每个域的掩码包含IB tokens
   ```python
   domain_masks[domain] = torch.cat([
       torch.ones(..., self.num_ib_tokens, ...),  # IB部分全为True
       domain_mask,  # 原始域掩码
   ], dim=1)
   ```

3. **层间IB同步** (第337-338行): 在指定的 `ib_comm_layers` 层进行跨域通信
   ```python
   if layer_index in self.ib_comm_layers:
       domain_embeddings = self.ib_token_sync(domain_embeddings)
   ```

4. **IB聚合** (`ib_token_sync` 第238-266行): 对各域的IB embedding做mean pooling并同步给所有域
   ```python
   aggregated_ib_embeddings += domain_embeddings[domain][:, :self.num_ib_tokens]
   aggregated_ib_embeddings /= len(domain_embeddings)  # 平均
   # 更新到所有域
   for domain in domain_embeddings:
       domain_embeddings[domain][:, :self.num_ib_tokens] = aggregated_ib_embeddings
   ```

---

### 5. 域嵌入合并
**代码位置:** `merge_domain_embeddings()` 第142-183行

```python
# in-place累加各域特定embedding
for domain in self.domain_encoders:
    mask = domain_masks[domain]
    embeddings[mask] += domain_specific_embeddings[domain][mask]

# LayerNorm归一化
embeddings = self.domain_fusion_layer_norm(embeddings)

# 可选：通过共享encoder进一步融合
if self.encoder:
    outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state
```

**分析:** 不同域的输出通过逐元素累加的方式合并，然后用LayerNorm稳定分布。如果有共享encoder，会进一步做cross-domain attention。

---

## 总结：域区分的三层机制

| 层级 | 机制 | 作用 |
|------|------|------|
| **输入层** | `domain_ids` 张量 | 显式标注每个token的域归属 |
| **掩码层** | `domain_masks` | 控制每个域encoder只能看到本域+mask+IB tokens |
| **参数层** | `domain_encoders` (ModuleDict) | 物理隔离，每个域有独立参数 |

**跨域通信渠道:** IB tokens (位置0~num_ib_tokens-1) 作为所有域共享的"信息瓶颈"，通过 `ib_token_sync` 在各层间同步梯度信息。
