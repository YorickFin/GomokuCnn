"""价值/策略网络。

修复要点：
1. policy 输出对非法着法（已占位置）做 mask，训练与推理保持一致；
2. 网络规模默认从 Config 读取（64 通道 / 6 残差块），并暴露
   hidden_channels / num_blocks 属性，便于自对弈子进程重建同构模型。
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config, board_size


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ValueCNN(nn.Module):
    """同时输出 value（tanh 标量）和 policy（225 维 logits）的网络。"""

    def __init__(self, in_channels=3, hidden_channels=None, num_blocks=None,
                 value_dim=256, value_conv_channels=4,
                 policy_mid_channels=None):
        super(ValueCNN, self).__init__()
        if hidden_channels is None:
            hidden_channels = Config.channel
        if num_blocks is None:
            num_blocks = Config.num_blocks
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks

        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3,
                                   padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        if policy_mid_channels is None:
            policy_mid_channels = hidden_channels
        self.policy_conv1 = nn.Conv2d(hidden_channels, policy_mid_channels,
                                      kernel_size=3, padding=1)
        self.policy_bn1 = nn.BatchNorm2d(policy_mid_channels)
        self.policy_conv2 = nn.Conv2d(policy_mid_channels, 1, kernel_size=3,
                                      padding=1)

        self.value_conv = nn.Conv2d(hidden_channels, value_conv_channels,
                                    kernel_size=1)
        # GroupNorm 替代 BatchNorm2d：单通道 BatchNorm 在某些 cudnn 版本
        #（如 Pascal 6.1 + cudnn 9.10）反向传播产生近零梯度，导致 value 头
        # 完全不学习。GroupNorm 数学等价但不依赖 cudnn，跨架构一致。
        self.value_norm = nn.GroupNorm(value_conv_channels,
                                       value_conv_channels)
        self.value_fc1 = nn.Linear(
            value_conv_channels * board_size * board_size, value_dim)
        self.value_fc2 = nn.Linear(value_dim, 1)

    def forward(self, x):
        # 非法着法 mask：第 3 通道（empty）为 0 的位置不允许落子
        empty = x[:, 2, :, :].view(x.size(0), -1)  # (B, 225)，1 表示空位
        legal_mask = empty > 0.5

        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy_logits = policy.view(x.size(0), -1)
        # 非法位置 logits 置为 dtype 可表示的最小值，softmax 后概率为 0
        # （fp32=-3.4e38 / fp16=-65504，经 softmax 均下溢为精确 0，
        #   与原 -1e9 数值等价；fp16 路径见优化方案.md 阶段 2）
        policy_logits = policy_logits.masked_fill(
            ~legal_mask, torch.finfo(policy_logits.dtype).min)

        value = F.relu(self.value_norm(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return value, policy_logits

    def calc(self, x):
        """推理接口：返回 value (B,1) 与 policy 概率 (B,15,15)。"""
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(x)
            probs = F.softmax(logits, dim=1).view(-1, board_size, board_size)
            return value, probs

    @classmethod
    def from_state_dict(cls, state_dict):
        """根据 state_dict 的权重形状推断结构并加载。

        用于从 checkpoint 加载结构可能不同的模型（评估、续训），
        避免硬编码默认结构导致 size mismatch。
        推断 hidden_channels / num_blocks / value_conv_channels / value_dim。
        """
        hidden_channels = state_dict["conv_init.weight"].shape[0]
        num_blocks = len({
            k.split(".")[1] for k in state_dict
            if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
        })
        value_conv_channels = state_dict["value_conv.weight"].shape[0]
        value_dim = state_dict["value_fc1.weight"].shape[0]
        policy_mid_channels = state_dict["policy_conv1.weight"].shape[0]
        model = cls(hidden_channels=hidden_channels, num_blocks=num_blocks,
                    value_dim=value_dim,
                    value_conv_channels=value_conv_channels,
                    policy_mid_channels=policy_mid_channels)
        model.load_state_dict(state_dict)
        return model


def load_model(path, device=None):
    """从 checkpoint 路径加载模型（自动推断结构），加载后置于评估模式。

    eval() 关闭 Dropout/BatchNorm 的训练行为；对弈/评估路径不得使用
    训练态。calc() 内部也会兜底 self.eval()，此处显式设置以防将来
    有调用方直接使用 forward。
    """
    if device is None:
        device = get_device()
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model = ValueCNN.from_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def board_to_tensor(board):
    """将 board_size x board_size 棋盘转换为 3 通道 tensor。

    board: 1 代表当前方，-1 代表对方，0 代表空。
    返回: (3, board_size, board_size) float tensor，
    通道顺序为 [当前玩家, 对手, 空位]。
    """
    board = np.array(list(board))
    current_player = (board == 1).astype(np.float32)
    opponent = (board == -1).astype(np.float32)
    empty = (board == 0).astype(np.float32)
    tensor = np.stack([current_player, opponent, empty], axis=0)
    return torch.FloatTensor(tensor)


def prepare_inference_model(model, channels_last):
    """按推理模式返回模型（阶段 2 计算侧优化，优化方案.md）。

    channels_last=True 时返回 NHWC 内存格式的模型副本：模型可能同时
    被训练循环共享（train.py 自对弈->训练交替），就地转换会让训练
    conv 也走 NHWC kernel，改变训练数值轨迹，故用副本隔离。
    """
    if not channels_last:
        return model
    import copy
    return copy.deepcopy(model).to(memory_format=torch.channels_last)


def calc_batch_arrays(model, boards, device, channels_last=False, fp16=False,
                      profile=None):
    """批量棋盘 (n,225) int8 → (values f64(n,), policies f64(n,225))。

    MCTS py / cpp 两个后端共用的推理路径（优化方案.md 阶段 2 要求
    走同一条 eval 路径）。基线（channels_last=False, fp16=False）与
    原 _make_eval_fn / get_calc_batch 数值逐位一致：编码为精确 0/1
    float32，同一 forward。
    fp16=True 时用 autocast 包住推理调用（仅 CUDA 生效），value 输出
    为 fp16 量化值、policy 经 softmax 仍为 fp32——数值路径变更，
    启用前须过固定评估集 loss + arena 验收。
    profile 非 None 时按批累积四段耗时（阶段 0-3.1 诊断，默认 None）：
    [编码, H2D, forward, D2H, 批数]，段间 cuda 硬同步仅诊断口径。
    """
    is_cuda = device.type == 'cuda'

    def _sync():
        if is_cuda and profile is not None:
            torch.cuda.synchronize(device)

    if profile is not None:
        _sync()
        t0 = time.perf_counter()
    n = boards.shape[0]
    b = boards.reshape(n, board_size, board_size)
    batch = np.stack([(b == 1).astype(np.float32),
                      (b == -1).astype(np.float32),
                      (b == 0).astype(np.float32)], axis=1)
    if profile is not None:
        _sync()
        t1 = time.perf_counter()
    tensor = torch.from_numpy(batch).to(device)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    if profile is not None:
        _sync()
        t2 = time.perf_counter()
    if fp16 and is_cuda:
        with torch.autocast('cuda', dtype=torch.float16):
            values, policies = model.calc(tensor)
    else:
        values, policies = model.calc(tensor)
    if profile is not None:
        _sync()
        t3 = time.perf_counter()
    v_np = values.detach().cpu().numpy().reshape(-1).astype(np.float64)
    p_np = (policies.detach().cpu().numpy().reshape(n, board_size * board_size)
            .astype(np.float64))
    if profile is not None:
        _sync()
        t4 = time.perf_counter()
        profile[0] += t1 - t0
        profile[1] += t2 - t1
        profile[2] += t3 - t2
        profile[3] += t4 - t3
        profile[4] += 1
    return v_np, p_np


def get_calc(model, board):
    """对单个棋盘做一次推理，返回 (value: float, policy: list[list[float]])。"""
    device = get_device()
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        value, policy = model.calc(board_tensor)
    return float(value), policy.squeeze(0).cpu().numpy().tolist()


def get_calc_batch(model, boards):
    """对一批棋盘做一次批量推理。

    返回 [(value: float, policy: list[list[float]]), ...]，顺序与 boards 一致。
    批量 forward 可摊薄 kernel 启动开销，是 MCTS 叶子评估的主要加速点。
    """
    if len(boards) == 0:
        return []
    device = get_device()
    batch = torch.stack([board_to_tensor(b) for b in boards], dim=0).to(device)
    with torch.no_grad():
        values, policies = model.calc(batch)
    values = values.detach().cpu().view(-1).numpy()
    policies = policies.detach().cpu().numpy()
    return [(float(values[i]), policies[i].tolist())
            for i in range(len(boards))]
