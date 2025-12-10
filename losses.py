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
    基于欧式距离的对比损失 (Margin-based Contrastive Loss)
    类似于 Hadsell et al., 2006
    """
    def __init__(self, margin=1.0, temperature=1.0):
        super().__init__()
        self.margin = margin
        # 注意：标准的Margin Loss通常不需要temperature，但如果你想保留缩放功能，可以设为1.0
        self.temperature = temperature 
    
    def forward(self, features, labels=None):
        """
        Args:
            features: shape [N, 2, D]
        """
        if len(features.shape) != 3:
            raise ValueError('ContrastiveLoss expects input shape [BatchSize, 2, Dim]')
            
        # [Fix 1] L2 归一化：这对对比学习至关重要，防止特征模长爆炸
        # features: [N, 2, D]
        features = F.normalize(features, p=2, dim=2)

        output1 = features[:, 0, :]
        output2 = features[:, 1, :]

        # 拼接: [2N, D]
        z = torch.cat([output1, output2], dim=0)
        N = output1.shape[0]
        
        # ========== 计算欧式距离矩阵 ==========
        # 使用 torch.cdist 计算距离比手写公式更稳定、更高效
        # z: [2N, D] -> distance_matrix: [2N, 2N]
        distance_matrix = torch.cdist(z, z, p=2)
        
        # 如果你坚持要用手写公式 (x^2 + y^2 - 2xy)，必须加 clamp 防止负数：
        # z_sq = torch.sum(z ** 2, dim=1, keepdim=True)
        # dist_sq = z_sq + z_sq.t() - 2 * torch.matmul(z, z.t())
        # dist_sq = torch.clamp(dist_sq, min=1e-8) # [Fix 2] 防止出现 -1e-18 导致的 NaN
        # distance_matrix = torch.sqrt(dist_sq)

        # 应用温度缩放 (可选，通常设为 1.0)
        distance_matrix = distance_matrix / self.temperature

        # ========== 构建掩码 ==========
        # 自身掩码
        mask_self = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        
        # 正样本掩码：只有来自同一个样本的两个视图互为正样本
        # (i, i+N) 和 (i+N, i) 是正样本
        mask_pos = torch.zeros_like(mask_self)
        mask_pos[:N, N:] = torch.eye(N, device=z.device, dtype=torch.bool)
        mask_pos[N:, :N] = torch.eye(N, device=z.device, dtype=torch.bool)
        
        # 负样本掩码：排除自身，排除正样本
        mask_neg = ~(mask_self | mask_pos)
        
        # ========== 计算损失 ==========
        # 1. 正样本损失: 最小化距离 (Pull together)
        # 提取正样本对的距离
        pos_dist = distance_matrix[mask_pos] # [2N]
        pos_loss = torch.mean(torch.pow(pos_dist, 2))

        # 2. 负样本损失: 距离需要大于 margin (Push apart)
        # 提取负样本对的距离
        neg_dist = distance_matrix[mask_neg] # [2N * (2N-2)]
        
        # Margin Logic: loss = max(0, margin - distance)^2
        neg_loss_raw = torch.clamp(self.margin - neg_dist, min=0.0)
        neg_loss = torch.mean(torch.pow(neg_loss_raw, 2))
        
        # 总损失
        total_loss = pos_loss + neg_loss
        
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
