"""LR 调度测试（2026-09-24 定稿：nn_012 方案）。

方案：train_model 每轮（每次调用）内部新建 plain Adam(lr=learning_rate)
+ ReduceLROnPlateau('min', patience, factor)，每个 epoch 用当轮验证集
总损失 val_total = 2*val_value + val_policy step 一次。优化器每轮重建 →
LR 每轮回 1e-4、Adam 动量不跨轮累积、scheduler 每轮重置；连续
patience 个 epoch 验证损失无改善则 lr × factor。

旧版"余弦下降 + 谷底停留 + arena 驱动跳回"状态机（LrController /
lr_state.txt）已全面弃用，相关测试一并删除。

验证：
1. Config 仅保留 nn_012 调度参数，旧状态机参数已删除；
2. train_model 签名不含 optimizer / scheduler（每轮内部新建）；
3. train_model 用 plain Adam（无 weight_decay）创建，lr=learning_rate；
4. 每次 train_model 调用重建优化器（跨轮不共享 Adam 动量）；
5. ReduceLROnPlateau 以 config.lr_patience / lr_factor 创建，且每
   epoch 用 val_total_loss 推进一次。
"""

import inspect

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import board_size, Config
from src.net import ValueCNN, get_device
from src.utils import WeightedDataset
import train


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
    return DataLoader(WeightedDataset(*data),
                     batch_size=batch_size, shuffle=shuffle)


class _SmallConfig:
    num_epochs = 3
    learning_rate = 1e-4
    lr_patience = 2
    lr_factor = 0.5
    train_amp_bf16 = False


# ---------------------------------------------------------------- Config

def test_config_params():
    """Config 仅保留 nn_012 调度参数；旧状态机参数应已删除。"""
    assert Config.learning_rate == 1e-4
    assert Config.lr_patience == 2
    assert Config.lr_factor == 0.5
    for stale in ('lr_min', 'lr_cycle_rounds', 'lr_jumpback_fails',
                  'weight_decay'):
        assert not hasattr(Config, stale), \
            f"旧参数 {stale} 应已删除（nn_012 方案不用）"


# ---------------------------------------------------------------- 签名

def test_train_model_signature_has_no_optimizer():
    """train_model 不得接收 optimizer / scheduler：每轮内部新建。"""
    params = inspect.signature(train.train_model).parameters
    assert 'optimizer' not in params, \
        "优化器每轮在 train_model 内部新建，不应再由外部传入"
    assert 'scheduler' not in params, \
        "ReduceLROnPlateau 每轮在 train_model 内部新建，不应由外部传入"
    assert 'config' in params


# ---------------------------------------------------------------- 优化器

class _SpyAdam(optim.Adam):
    """记录构造参数的真实 Adam 子类（仍可作为优化器正常工作）。"""
    constructed = []

    def __init__(self, params, lr, **kwargs):
        _SpyAdam.constructed.append({'lr': lr, 'kwargs': dict(kwargs)})
        super().__init__(params, lr=lr, **kwargs)

    @classmethod
    def reset(cls):
        cls.constructed = []


@pytest.fixture(autouse=True)
def _reset_spy():
    _SpyAdam.reset()
    _SpyPlateau.reset()
    yield
    _SpyAdam.reset()
    _SpyPlateau.reset()


def test_optimizer_is_plain_adam_no_weight_decay(monkeypatch):
    """train_model 创建 plain Adam：lr=learning_rate，无 weight_decay。"""
    monkeypatch.setattr(train.optim, 'Adam', _SpyAdam)
    device = get_device()
    model = make_model().to(device)
    loader = make_loader(make_fake_data(16), shuffle=True)

    train.train_model(model, loader, loader, _SmallConfig())

    assert len(_SpyAdam.constructed) == 1
    rec = _SpyAdam.constructed[0]
    assert rec['lr'] == _SmallConfig.learning_rate
    assert 'weight_decay' not in rec['kwargs'], \
        "nn_012 方案用 plain Adam，不应传 weight_decay"


def test_optimizer_recreated_each_call(monkeypatch):
    """每次 train_model 调用都新建优化器：Adam 动量不跨轮累积。"""
    monkeypatch.setattr(train.optim, 'Adam', _SpyAdam)
    device = get_device()
    model = make_model().to(device)
    loader = make_loader(make_fake_data(16), shuffle=True)

    for _ in range(3):
        train.train_model(model, loader, loader, _SmallConfig())

    # 3 次调用 → 3 次新建（每轮重建，动量不跨轮共享）
    assert len(_SpyAdam.constructed) == 3
    for rec in _SpyAdam.constructed:
        assert rec['lr'] == _SmallConfig.learning_rate


# ---------------------------------------------------------------- 调度器

class _SpyPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """记录构造参数与 step 调用的真实 ReduceLROnPlateau 子类。"""
    constructed = []
    steps = []

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 **kwargs):
        _SpyPlateau.constructed.append(
            {'mode': mode, 'factor': factor, 'patience': patience})
        super().__init__(optimizer, mode=mode, factor=factor,
                        patience=patience, **kwargs)

    def step(self, metrics=None):
        _SpyPlateau.steps.append(metrics)
        super().step(metrics)

    @classmethod
    def reset(cls):
        cls.constructed = []
        cls.steps = []


def test_scheduler_created_with_config_params(monkeypatch):
    """ReduceLROnPlateau 以 config.lr_patience / lr_factor 创建。"""
    monkeypatch.setattr(train.optim, 'Adam', _SpyAdam)
    monkeypatch.setattr(train.optim.lr_scheduler, 'ReduceLROnPlateau',
                        _SpyPlateau)
    device = get_device()
    model = make_model().to(device)
    loader = make_loader(make_fake_data(16), shuffle=True)

    train.train_model(model, loader, loader, _SmallConfig())

    assert len(_SpyPlateau.constructed) == 1
    rec = _SpyPlateau.constructed[0]
    assert rec['mode'] == 'min'
    assert rec['patience'] == _SmallConfig.lr_patience
    assert rec['factor'] == _SmallConfig.lr_factor


def test_scheduler_steps_once_per_epoch_on_val_total(monkeypatch):
    """每个 epoch 用当轮验证总损失 val_total 推进一次 scheduler。"""
    monkeypatch.setattr(train.optim, 'Adam', _SpyAdam)
    monkeypatch.setattr(train.optim.lr_scheduler, 'ReduceLROnPlateau',
                        _SpyPlateau)
    device = get_device()
    model = make_model().to(device)
    loader = make_loader(make_fake_data(16), shuffle=True)

    train.train_model(model, loader, loader, _SmallConfig())

    n_ep = _SmallConfig.num_epochs
    assert len(_SpyPlateau.steps) == n_ep
    for m in _SpyPlateau.steps:
        # val_total_loss = 2*val_value + val_policy，恒为正有限数
        assert m is not None and m > 0 and m == m
