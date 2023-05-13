import torch
import torch.nn.functional as F


# 注意：所有q都不计算梯度  为什么要双重detach()？？直接advantages.detach()不就行了吗？？
def compute_baseline(pi_probs, qvalues):
    ''' baseline = ∑ π(s,b,θ)q(s,b,w) '''                      # multiply:对应元素相承
    baseline = torch.sum(torch.multiply(pi_probs, qvalues.detach()), dim=1)       # (batch,1)
    return baseline


def compute_advantages(pi_probs, qvalues, use_relu=False):
    baseline = compute_baseline(pi_probs, qvalues.detach())
    ''' advantages = q(s,a,w) - ∑ π(s,b,θ)q(s,b,w) '''            # 注意：advantages整体都不计算梯度
    advantages = qvalues.detach() - baseline.reshape(baseline.shape[0], -1)
    if use_relu == True:
        advantages = F.relu(advantages)
    ''' policy_advantages = ∑ π(s,a,θ)[q(s,a,w) - ∑ π(s,b,θ)q(s,b,w)]  且只对π(s,a,θ)求梯度'''
    policy_advantages = torch.sum(torch.multiply(pi_probs, advantages.detach()), dim=1)      # todo 注意求和维度
    return policy_advantages


def compute_regrets(pi_probs, qvalues, use_relu=True):                  # 默认使用relu
    baseline = compute_baseline(pi_probs, qvalues.detach())
    '''regrets = ∑ [q(s,a,w) - ∑ π(s,b,θ)q(s,b,w)]'''
    regrets = torch.sum(F.relu(qvalues.detach() - baseline.reshape(baseline.shape[0], -1)), dim=1)      # qvalues:(batch,2) baseline:(batch,1) 广播？？
    return regrets

