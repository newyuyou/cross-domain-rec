from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import SelfAttention, CrossAttentionNoMask
from models.ffn import FeedForward

from models.utils.initialization import init_weights
from models.utils.position import get_absolute_pos_idx
from models.data.evaluation import cal_norm_mask


class DomainDualTower(nn.Module):
    """
    Dual-tower architecture for each domain:
    - Pure tower: causal self-attention only (strictly autoregressive on own domain)
    - Mixed tower: causal self-attention + cross-attention (no mask) to other domain
    """
    def __init__(self,
                 d_embed: int,
                 n_head: int,
                 len_trim: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Pure tower: strictly causal, only sees own domain history
        self.sa_pure = SelfAttention(d_embed, n_head, len_trim, dropout)
        self.ffn_pure = FeedForward(d_embed, dropout)

        # Mixed tower: causal self-attention + no-mask cross-attention
        self.sa_mixed = SelfAttention(d_embed, n_head, len_trim, dropout)
        self.ca_mixed = CrossAttentionNoMask(d_embed, n_head, dropout)
        self.ffn_mixed = FeedForward(d_embed, dropout)

    def forward_pure(self,
                     x: torch.Tensor,
                     mask: torch.Tensor,
                     ) -> torch.Tensor:
        """
        Pure tower forward pass.

        Args:
            x: input embeddings (batch, len, dim)
            mask: domain mask (batch, len, 1)

        Returns:
            h_pure: pure tower output (batch, len, dim)
        """
        h_pure = self.sa_pure(x, mask)
        h_pure = self.ffn_pure(h_pure, mask)
        return h_pure

    def forward_mixed(self,
                      x: torch.Tensor,
                      mask: torch.Tensor,
                      other_pure: torch.Tensor,
                      ) -> torch.Tensor:
        """
        Mixed tower forward pass.

        Args:
            x: input embeddings (batch, len, dim)
            mask: domain mask (batch, len, 1)
            other_pure: other domain's pure tower output (batch, len, dim)

        Returns:
            h_mixed: mixed tower output (batch, len, dim)
        """
        # Self-attention with causal mask (own domain history)
        h_mixed = self.sa_mixed(x, mask)

        # Cross-attention without mask (can see full other domain sequence)
        h_mixed = self.ca_mixed(h_mixed, other_pure, mask)

        # FFN
        h_mixed = self.ffn_mixed(h_mixed, mask)

        return h_mixed

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                other_pure: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through both towers.

        Args:
            x: input embeddings (batch, len, dim)
            mask: domain mask (batch, len, 1)
            other_pure: other domain's pure tower output (batch, len, dim)

        Returns:
            h_pure: pure tower output (batch, len, dim)
            h_mixed: mixed tower output (batch, len, dim)
        """
        h_pure = self.forward_pure(x, mask)
        h_mixed = self.forward_mixed(x, mask, other_pure)
        return h_pure, h_mixed


class MERITDual(nn.Module):
    """
    MERIT with Dual-Tower Architecture for Cross-Domain Recommendation.

    Each domain has two towers:
    - Pure tower: strictly autoregressive on own domain
    - Mixed tower: autoregressive on own domain + full access to other domain

    Key difference from original MERIT:
    - Cross-attention in mixed towers has NO causal mask,
      allowing access to full cross-domain information.
    """
    def __init__(self,
                 args: Namespace,
                 ) -> None:
        super().__init__()
        self.bs = args.bs
        self.len_trim: int = args.len_trim
        self.n_item: int = args.n_item
        self.n_item_a: int = args.n_item_a
        self.n_neg: int = args.n_neg
        self.temp: float = args.temp
        self.d_embed: int = args.d_embed

        self.dropout = nn.Dropout(args.dropout) if args.dropout > 0 else nn.Identity()

        # Embeddings (shared across domains)
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)

        # Dual towers for each domain
        self.tower_a = DomainDualTower(args.d_embed, args.n_head, args.len_trim, args.dropout)
        self.tower_b = DomainDualTower(args.d_embed, args.n_head, args.len_trim, args.dropout)

        # Learnable fusion gate for combining pure and mixed representations
        # Strategy 2: learnable weighted fusion
        self.fusion_gate_a = nn.Linear(self.d_embed * 2, 1)
        self.fusion_gate_b = nn.Linear(self.d_embed * 2, 1)

        self.apply(init_weights)

    def embed_pos(self,
                  mask: torch.Tensor,
                  ) -> torch.Tensor:
        """Get positional embeddings based on mask."""
        return self.ep(get_absolute_pos_idx(mask))

    def forward(self,
                seq_m: torch.Tensor,
                idx_last_a: torch.Tensor = None,
                idx_last_b: torch.Tensor = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual-tower architecture.

        Args:
            seq_m: mixed sequence with items from both domains
            idx_last_a: last position indices for domain A (inference only)
            idx_last_b: last position indices for domain B (inference only)

        Returns:
            h_pred_a: fused representation for domain A prediction
            h_pred_b: fused representation for domain B prediction
            h_pure_a, h_pure_b: pure tower outputs
            h_mixed_a, h_mixed_b: mixed tower outputs
        """
        # Create masks
        # mask_a: domain A items (seq_m > n_item_a) - note: in original code this seems reversed
        # mask_b: domain B items (0 < seq_m <= n_item_a)
        mask_m = (seq_m != 0).to(torch.int32).unsqueeze(-1)
        mask_a = (seq_m > self.n_item_a).to(torch.int32).unsqueeze(-1)  # Domain A mask
        mask_b = ((seq_m > 0) & (seq_m <= self.n_item_a)).to(torch.int32).unsqueeze(-1)  # Domain B mask

        # Embeddings + positional encoding
        h_m = self.ei(seq_m)
        h_a = self.dropout(h_m + self.embed_pos(mask_m)) * mask_a  # Domain A sequence
        h_b = self.dropout(h_m + self.embed_pos(mask_m)) * mask_b  # Domain B sequence

        # Step 1: Get pure tower outputs first (no cross-domain dependency)
        h_pure_a = self.tower_a.forward_pure(h_a, mask_a)
        h_pure_b = self.tower_b.forward_pure(h_b, mask_b)

        # Step 2: Get mixed tower outputs (cross-attention to other domain's pure output)
        h_mixed_a = self.tower_a.forward_mixed(h_a, mask_a, h_pure_b)
        h_mixed_b = self.tower_b.forward_mixed(h_b, mask_b, h_pure_a)

        # Step 3: Fuse pure and mixed representations
        # Strategy 1: Simple addition (default, more stable)
        h_pred_a = h_pure_a + h_mixed_a
        h_pred_b = h_pure_b + h_mixed_b

        # Strategy 2: Learnable gating (uncomment if needed)
        # gate_a = torch.sigmoid(self.fusion_gate_a(torch.cat([h_pure_a, h_mixed_a], dim=-1)))
        # gate_b = torch.sigmoid(self.fusion_gate_b(torch.cat([h_pure_b, h_mixed_b], dim=-1)))
        # h_pred_a = gate_a * h_pure_a + (1 - gate_a) * h_mixed_a
        # h_pred_b = gate_b * h_pure_b + (1 - gate_b) * h_mixed_b

        # Inference: extract representations at last positions
        if not self.training:
            idx_batched = torch.arange(h_pred_a.size(0))

            # Extract at specific positions for each domain
            h_pred_a = h_pred_a[idx_batched, idx_last_a.squeeze(-1)]
            h_pred_b = h_pred_b[idx_batched, idx_last_b.squeeze(-1)]

            # Also return pure and mixed for analysis
            h_pure_a = h_pure_a[idx_batched, idx_last_a.squeeze(-1)]
            h_pure_b = h_pure_b[idx_batched, idx_last_b.squeeze(-1)]
            h_mixed_a = h_mixed_a[idx_batched, idx_last_a.squeeze(-1)]
            h_mixed_b = h_mixed_b[idx_batched, idx_last_b.squeeze(-1)]

        return h_pred_a, h_pred_b, h_pure_a, h_pure_b, h_mixed_a, h_mixed_b

    def cal_rec_loss(self,
                     h: torch.Tensor,
                     gt: torch.Tensor,
                     gt_neg: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate recommendation loss using InfoNCE.

        Args:
            h: predicted representations (batch, len, dim) or (batch, dim)
            gt: ground truth item indices (batch, len) or (batch,)
            gt_neg: negative samples (batch, len, n_neg) or (batch, n_neg)

        Returns:
            loss_a: loss for domain A
            loss_b: loss for domain B
        """
        # Create domain masks for ground truth
        mask_gt_a = torch.where(gt.gt(0) & gt.le(self.n_item_a), 1, 0)
        mask_gt_b = torch.where(gt.gt(self.n_item_a), 1, 0)

        # Get embeddings
        e_gt = self.ei(gt).unsqueeze(-2)
        e_neg = self.ei(gt_neg)
        e_all = torch.cat([e_gt, e_neg], dim=-2)

        # Calculate logits with temperature scaling
        logits = torch.einsum('bld,blnd->bln', h, e_all).div(self.temp)

        # InfoNCE loss
        loss = -F.log_softmax(logits, dim=2)[:, :, 0]

        # Domain-specific losses (normalized by sequence length)
        loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()

        return loss_a, loss_b

    def cal_rank(self,
                 h_pred_a: torch.Tensor,
                 h_pred_b: torch.Tensor,
                 gt: torch.Tensor,
                 gt_mtc: torch.Tensor,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate ranking via inner-product similarity.

        Args:
            h_pred_a: fused representation for domain A (batch, dim)
            h_pred_b: fused representation for domain B (batch, dim)
            gt: ground truth item indices (batch, 1)
            gt_mtc: metric candidate items (batch, n_mtc)

        Returns:
            ranks: rank of ground truth (batch,)
            mask_gt_a: domain A mask for ground truth (batch, 1)
            mask_gt_b: domain B mask for ground truth (batch, 1)
        """
        # Domain masks
        mask_gt_a = torch.where(gt <= self.n_item_a, 1, 0)
        mask_gt_b = torch.where(gt > self.n_item_a, 1, 0)

        # Select appropriate representation based on domain
        h = h_pred_a * mask_gt_a + h_pred_b * mask_gt_b

        # Get item embeddings
        e_gt, e_mtc = self.ei(gt), self.ei(gt_mtc)
        e_all = torch.cat([e_gt, e_mtc], dim=1)

        # Calculate similarity scores
        logits = torch.einsum('bd,bnd->bn', h, e_all)

        # Rank: number of items scored higher than ground truth + 1
        ranks = logits[:, 1:].gt(logits[:, 0:1]).sum(dim=1).add(1)

        return ranks, mask_gt_a, mask_gt_b
