"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    无label的对比损失类（仿照SimCLR损失的实现逻辑）
    适配场景：输入为(x1, x2)，其中x1和x2是同一样本的两个增强视图（正样本对）
    无需手动传入label，自动构建正/负样本对
    """
    def __init__(self, margin=2.0, temperature=0.1):
        super().__init__()
        self.margin = margin  # 对比损失的边际值
        self.temperature = temperature  # 可选：缩放距离，提升区分度
    
    def forward(self, output1, output2):
        """
        Args:
            output1: 孪生网络分支1输出，shape [N, D]（N=批次大小，D=特征维度）
            output2: 孪生网络分支2输出，shape [N, D]
        Returns:
            批次平均对比损失
        """
        # ========== 步骤1：拼接特征，构建批次内所有样本对 ==========
        # 拼接output1和output2，形成 [2N, D]（和SimCLR的2N个增强视图一致）
        z = torch.cat([output1, output2], dim=0)  # [2N, D]
        N = output1.shape[0]
        
        # ========== 步骤2：计算所有样本对的欧式距离 ==========
        # 计算两两欧式距离：shape [2N, 2N]
        # 方法：distance(i,j) = sqrt(||z[i] - z[j]||²)
        z_sq = torch.sum(z **2, dim=1, keepdim=True)  # [2N, 1]
        distance_matrix = torch.sqrt(
            z_sq + z_sq.t() - 2 * torch.matmul(z, z.t()) + 1e-8  # 加小值避免根号内为负
        ) / self.temperature  # 温度缩放，提升区分度
        
        # ========== 步骤3：构建正/负样本掩码（无需手动label） ==========
        # 自身掩码：屏蔽样本自身的距离（distance(i,i)=0）
        mask_self = torch.eye(2*N, device=z.device, dtype=torch.bool)
        # 正样本掩码：(0,N), (1,N+1), ..., (N-1,2N-1) 和 (N,0), (N+1,1), ..., (2N-1,N-1)
        mask_pos = torch.zeros_like(mask_self)
        mask_pos[:N, N:] = torch.eye(N, device=z.device, dtype=torch.bool)
        mask_pos[N:, :N] = torch.eye(N, device=z.device, dtype=torch.bool)
        # 负样本掩码：非自身、非正样本的所有样本对
        mask_neg = ~(mask_self | mask_pos)
        
        # ========== 步骤4：计算对比损失 ==========
        # 提取正样本对的距离
        pos_dist = distance_matrix[mask_pos].view(-1, 1)  # [2N, 1]
        # 提取负样本对的距离（每个正样本对匹配所有负样本对）
        neg_dist = distance_matrix[mask_neg].view(2*N, -1)  # [2N, 2N-2]
        
        # 经典对比损失公式：
        # 正样本对损失：(1-label) * distance² → label=0，即 pos_dist²
        # 负样本对损失：label * max(margin - distance, 0)² → label=1，即 max(margin - neg_dist, 0)²
        pos_loss = torch.mean(torch.pow(pos_dist, 2))  # 正样本对：最小化距离
        neg_loss = torch.mean(torch.pow(torch.clamp(self.margin - neg_dist, min=0.0), 2))  # 负样本对：距离≥margin
        
        # 总损失：正样本损失 + 负样本损失（平均）
        total_loss = (pos_loss + neg_loss) / 2
        
        return total_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=mask_pos_pairs.device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
