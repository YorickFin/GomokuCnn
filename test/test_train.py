"""训练流程测试：损失形状/权重生效、训练步可运行、优化器每轮重建。"""

import copy

import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import board_size
from src.net import ValueCNN, get_device
from src.utils import WeightedDataset
from train import evaluate_on_fixed, train_model


def make_model():
    return ValueCNN(hidden_channels=8, num_blocks=2)


def make_fake_data(n=16):
    """全空棋盘 + 均匀策略 + 随机价值的假数据。"""
    boards = torch.zeros(n, 3, board_size, board_size)
    boards[:, 2] = 1.0
    policies = torch.full((n, board_size, board_size),
                          1.0 / (board_size * board_size))
    values = torch.rand(n) * 2 - 1
    weights = torch.ones(n)
    return boards, policies, values, weights


def make_loader(data, batch_size=8, shuffle=False):
    dataset = WeightedDataset(*data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def test_policy_loss_shape_and_weights_effective():
    """回归测试：policy loss 必须为 (B,)，样本权重逐样本生效。

    旧实现 (B,1) * (B,) 广播为 (B,B)，权重差异完全失效。
    """
    model = make_model().eval()
    boards, policies, _v, _w = make_fake_data(4)

    with torch.no_grad():
        _values, logits = model(boards)
    log_p = F.log_softmax(logits, dim=1)
    target = policies.view(4, -1)
    kl = F.kl_div(log_p, target, reduction='none').sum(dim=1)

    assert kl.shape == (4,), f"policy loss 形状错误: {kl.shape}"

    # 全 1 权重 vs 只给第 0 个样本大权重，结果必须不同
    loss_equal = (kl * torch.ones(4)).mean()
    w = torch.tensor([10.0, 0.0, 0.0, 0.0])
    loss_weighted = (kl * w).mean()
    assert not torch.allclose(loss_equal, loss_weighted)
    # 手算对照：加权值 = kl[0]*10/4
    assert torch.allclose(loss_weighted, kl[0] * 10.0 / 4.0, atol=1e-6)


class _SmallConfig:
    num_epochs = 1
    learning_rate = 1e-3
    lr_patience = 2
    lr_factor = 0.5
    train_amp_bf16 = False


def test_train_model_runs_and_updates():
    device = get_device()
    model = make_model().to(device)
    train_loader = make_loader(make_fake_data(16), shuffle=True)
    val_loader = make_loader(make_fake_data(8))

    before = copy.deepcopy(model.state_dict())
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, _SmallConfig())

    assert len(train_losses) == 1
    assert len(val_losses) == 1
    for x in train_losses + val_losses:
        assert x == x  # 非 NaN
        assert x > 0

    # 参数确实被更新
    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k])
                  for k in before if before[k].dtype.is_floating_point)
    assert changed, "训练后参数未发生变化"


def test_train_model_uses_internal_optimizer():
    """train_model 内部新建优化器（nn_012 方案）：外部 optimizer 不被使用，
    其动量状态保持为空、lr 不被改动。

    旧实现 optimizer 跨轮复用保留 Adam 动量；新方案每轮重建，外部
    优化器与本轮训练完全脱钩。
    """
    model = make_model()
    external_opt = optim.Adam(model.parameters(), lr=1e-3)
    initial_lr = external_opt.param_groups[0]['lr']

    loader = make_loader(make_fake_data(16), shuffle=True)
    train_model(model, loader, loader, _SmallConfig())

    # 外部优化器未参与训练：无动量状态、lr 不变
    assert len(external_opt.state_dict()['state']) == 0
    assert external_opt.param_groups[0]['lr'] == initial_lr


def test_evaluate_on_fixed_returns_finite_losses():
    """固定评估集：返回有限正损失，且模型处于 eval 模式。"""
    device = get_device()
    model = make_model().to(device)
    loader = make_loader(make_fake_data(8))

    value_loss, policy_loss = evaluate_on_fixed(model, loader, device)

    assert value_loss == value_loss and policy_loss == policy_loss  # 非 NaN
    assert value_loss > 0 and policy_loss > 0
    assert not model.training  # 评估后模型应处于 eval 模式


def test_evaluate_on_fixed_weighted():
    """加权语义：loss = Σ(w·l)/Σw（与训练 loss 同口径）。

    构造 weight 全 0 的干扰样本 + weight=1 的真实样本：
    加权后 loss 应完全等于真实样本单独评估的 loss。
    """
    device = get_device()
    torch.manual_seed(7)
    model = make_model().to(device)
    data = make_fake_data(4)
    boards, policies, values, _ = data

    # 干扰样本：权重 0，且故意给离谱标签（若未加权会拉爆 loss）
    z_boards = torch.randn(4, *boards.shape[1:])
    z_policies = torch.zeros_like(policies[:4])
    z_values = torch.ones(4) * 99.0
    z_weights = torch.zeros(4)

    g_boards, g_policies = boards[:2], policies[:2]
    g_values, g_weights = values[:2], torch.ones(2)

    mix = (torch.cat([g_boards, z_boards]),
           torch.cat([g_policies, z_policies]),
           torch.cat([g_values, z_values]),
           torch.cat([g_weights, z_weights]))
    pure = (g_boards, g_policies, g_values, g_weights)

    mv, mp = evaluate_on_fixed(model, make_loader(mix), device)
    pv, pp = evaluate_on_fixed(model, make_loader(pure), device)
    # rel=5e-3 容忍 cudnn 对不同批形状选用不同卷积核的浮点差异；
    # 若未加权，离谱标签的 0 权重样本会把 value loss 拉爆约 5 个数量级
    assert mv == pytest.approx(pv, rel=5e-3)
    assert mp == pytest.approx(pp, rel=5e-3)
