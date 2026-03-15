from argparse import Namespace
import time
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from models.DualDecoder import MERITDual
from models.data.evaluation import cal_metrics
from models.data.dataloader import get_dataloader

from noter import Noter


class TrainerDual(object):
    """
    Trainer for MERITDual model with dual-tower architecture.

    Key differences from original Trainer:
    - Uses MERITDual model instead of MERIT
    - Simplified loss calculation (no mixed sequence loss)
    - Direct use of fused representations for prediction
    """
    def __init__(self,
                 args: Namespace,
                 noter: Noter
                 ) -> None:
        print('[info] Loading data')
        self.n_warmup = args.n_warmup
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.n_user = args.n_user
        self.n_item_a = args.n_item_a
        print('Done.\n')

        self.noter = noter
        self.device = args.device

        # Model: MERITDual with dual-tower architecture
        self.model = MERITDual(args).to(args.device)

        # Optimizer and schedulers
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1., total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p)

        noter.log_num_param(self.model)

    def run_epoch(self,
                  i_epoch: int,
                  ) -> tuple[list | None, list | None]:
        """
        Run one training epoch and validation.

        Args:
            i_epoch: current epoch index

        Returns:
            metrics for domain A and domain B (None during warmup)
        """
        self.model.train()
        loss_a, loss_b = 0., 0.
        t_0 = time.time()

        # Training
        for batch in tqdm(self.train_loader, desc='training', leave=False):
            self.optimizer.zero_grad()

            loss_a_batch, loss_b_batch = self.train_batch(batch)

            n_seq = batch[0].size(0)
            loss_a += (loss_a_batch * n_seq) / self.n_user
            loss_b += (loss_b_batch * n_seq) / self.n_user

        # Log training loss (use 0 for loss_m to maintain compatibility)
        self.noter.log_train(i_epoch, loss_a, loss_b, 0., time.time() - t_0)

        # Warm-up quit
        if i_epoch <= self.n_warmup:
            return None, None

        # Validation
        self.model.eval()
        ranks_f2a, ranks_f2b = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='validating', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def run_test(self,
                 ) -> tuple[list, list]:
        """
        Evaluate on test set.

        Returns:
            metrics for domain A and domain B
        """
        self.model.eval()
        ranks_f2a, ranks_f2b = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='testing', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def train_batch(self,
                    batch: list[torch.Tensor],
                    ) -> tuple[float, float]:
        """
        Process a single training batch.

        Args:
            batch: list of tensors [seq_m, gt_m, gt_ab, gt_neg_m, gt_neg_ab]

        Returns:
            loss_a, loss_b for domain A and B
        """
        # seq_m: mixed sequence, gt_m: ground truth for mixed, gt_ab: ground truth for domains
        # We use gt_ab for domain-specific prediction
        seq_m, gt_m, gt_ab, gt_neg_m, gt_neg_ab = map(lambda x: x.to(self.device), batch)

        # Forward pass through dual-tower model
        # Returns: h_pred_a, h_pred_b, h_pure_a, h_pure_b, h_mixed_a, h_mixed_b
        h_pred_a, h_pred_b, _, _, _, _ = self.model(seq_m)

        # Calculate losses for each domain using fused representations
        # Domain A loss
        loss_a_a, loss_a_b = self.model.cal_rec_loss(h_pred_a, gt_ab, gt_neg_ab)
        # Domain B loss
        loss_b_a, loss_b_b = self.model.cal_rec_loss(h_pred_b, gt_ab, gt_neg_ab)

        # Total loss: sum of domain-specific losses
        # Note: loss_a_b and loss_b_a should be 0 (wrong domain)
        loss_a_total = loss_a_a + loss_a_b
        loss_b_total = loss_b_a + loss_b_b

        (loss_a_total + loss_b_total).backward()
        self.optimizer.step()

        return loss_a_total.item(), loss_b_total.item()

    def evaluate_batch(self,
                       batch: list[torch.Tensor],
                       ) -> tuple[list[float], list[float]]:
        """
        Process a single evaluation batch.

        Args:
            batch: list of tensors [seq_m, idx_last_a, idx_last_b, gt, gt_mtc]

        Returns:
            ranks_a, ranks_b: lists of ranks for each domain
        """
        seq_m, idx_last_a, idx_last_b, gt, gt_mtc = map(lambda x: x.to(self.device), batch)

        # Forward pass with last position indices
        h_pred_a, h_pred_b, _, _, _, _ = self.model(seq_m, idx_last_a, idx_last_b)

        # Calculate ranks
        ranks, mask_gt_a, mask_gt_b = self.model.cal_rank(h_pred_a, h_pred_b, gt, gt_mtc)

        # Separate ranks by domain
        ranks_a = ranks[mask_gt_a.squeeze(-1) == 1].tolist()
        ranks_b = ranks[mask_gt_b.squeeze(-1) == 1].tolist()

        return ranks_a, ranks_b
