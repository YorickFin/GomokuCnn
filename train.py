"""自对弈训练入口。

LR 调度采用 nn_012 方案（2026-09-24 定稿）：train_model 每轮内部
新建 optim.Adam(lr=learning_rate) + ReduceLROnPlateau('min',
patience, factor)，每个 epoch 用当轮验证集总损失
val_total = 2*val_value + val_policy step 一次。优化器每轮重建 →
LR 每轮回 1e-4、Adam 动量不跨轮累积、scheduler 每轮重置；连续
patience 个 epoch 验证损失无改善则 lr × factor。旧版"余弦下降 +
谷底停留 + arena 驱动跳回"状态机（LrController / lr_state.txt）
已全面弃用。

其余要点：
1. policy 损失压缩为 (B,) 后再与样本权重相乘，修复原 (B,1)*(B,)
   广播成 (B,B) 导致样本权重失效的 bug；
2. 验证集按局切分（见 src/utils.py），val loss 可信；
3. 训练过程统一使用 logging 记录（文件 + 控制台），不使用 print。
"""

import logging
import multiprocessing
import os
from collections import deque

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config
from src.eval import pit
from src.logger import setup_logging
from src.net import ValueCNN, get_device
from src.utils import (WeightedDataset, _generate_games, augment_data,
                       concat_games, generate_fixed_eval_set,
                       load_fixed_eval_set, save_fixed_eval_set,
                       split_games)

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, config):
    """训练若干 epoch，返回 (train_losses, val_losses)。

    nn_012 LR 方案：本函数每轮（每次调用）内部新建优化器与调度器，
    优化器不跨轮复用、Adam 动量不跨轮累积。LR 起点恒为
    config.learning_rate，由 ReduceLROnPlateau 在轮内按验证损失动态下调。

    性能（2026-09-23 基准，test/test_benchmark_train.py）：
    forward 用 bf16 autocast（loss 在 autocast 外以 fp32 计算，bf16 与
    fp32 同 dynamic range 无需 GradScaler），输入/模型走 channels_last
    ——两项捆绑拿到 3.64x（GPU 段 328→102s）。config.train_amp_bf16
    可一键回退 fp32 路径。
    """
    device = get_device()
    use_amp = bool(getattr(config, "train_amp_bf16", False)) \
        and device.type == 'cuda'
    # channels_last 仅与 bf16 捆绑启用（fp32 下单独开是 2 倍负优化）
    mem_fmt = torch.channels_last if use_amp else torch.contiguous_format
    model.to(device=device, memory_format=mem_fmt)

    # ---- 每轮重建优化器 + 调度器（nn_012 方案）----
    # plain Adam 无 weight_decay；LR 起点 learning_rate，连续 lr_patience
    # 个 epoch 验证总损失无改善则 lr × lr_factor。
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=getattr(config, 'lr_patience', 2),
        factor=getattr(config, 'lr_factor', 0.5))

    def _forward(boards):
        if use_amp:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pv, pp = model(boards)
            return pv.float(), pp.float()
        return model(boards)

    value_criterion = nn.MSELoss(reduction='none')
    policy_criterion = nn.KLDivLoss(reduction='none')
    val_value_criterion = nn.MSELoss()
    val_policy_criterion = nn.KLDivLoss(reduction='batchmean')

    train_losses = []
    val_losses = []

    for epoch in range(config.num_epochs):
        # ---- 训练 ----
        model.train()
        train_value_loss = 0.0
        train_policy_loss = 0.0

        for batch_boards, batch_policies, batch_values, batch_weights in train_loader:
            batch_boards = batch_boards.to(device, memory_format=mem_fmt)
            batch_policies = batch_policies.to(device).view(
                batch_policies.size(0), -1)
            batch_values = batch_values.to(device).unsqueeze(1)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()
            pred_values, pred_policies = _forward(batch_boards)

            # value loss: (B,1) -> (B,)
            value_loss = value_criterion(pred_values, batch_values).squeeze(1)
            # policy loss: KL 对 225 维求和 -> (B,)，修复广播 bug
            policy_loss = policy_criterion(
                F.log_softmax(pred_policies, dim=1),
                batch_policies
            ).sum(dim=1)

            weighted_value_loss = (value_loss * batch_weights).mean()
            weighted_policy_loss = (policy_loss * batch_weights).mean()
            loss = 2 * weighted_value_loss + weighted_policy_loss

            loss.backward()
            optimizer.step()

            train_value_loss += weighted_value_loss.item()
            train_policy_loss += weighted_policy_loss.item()

        # ---- 验证 ----
        model.eval()
        val_value_loss = 0.0
        val_policy_loss = 0.0
        with torch.no_grad():
            for boards, policies, values, _weights in val_loader:
                boards = boards.to(device, memory_format=mem_fmt)
                policies = policies.to(device).view(policies.size(0), -1)
                values = values.to(device).unsqueeze(1)

                pred_values, pred_policies = _forward(boards)
                val_value_loss += val_value_criterion(
                    pred_values, values).item()
                val_policy_loss += val_policy_criterion(
                    F.log_softmax(pred_policies, dim=1),
                    policies
                ).item()

        avg_train_value = train_value_loss / max(len(train_loader), 1)
        avg_train_policy = train_policy_loss / max(len(train_loader), 1)
        avg_val_value = val_value_loss / max(len(val_loader), 1)
        avg_val_policy = val_policy_loss / max(len(val_loader), 1)

        logger.info(
            "[epoch %d] train value=%.4f policy=%.4f | "
            "val value=%.4f policy=%.4f | lr=%.2e",
            epoch, avg_train_value, avg_train_policy,
            avg_val_value, avg_val_policy,
            optimizer.param_groups[0]['lr'])

        val_total_loss = 2 * avg_val_value + avg_val_policy
        # ReduceLROnPlateau：当轮验证损失驱动，连续 patience 个 epoch
        # 无改善则降 lr（轮内调度，与 arena / 固定评估集解耦）
        scheduler.step(val_total_loss)

        train_losses.append(avg_train_value + avg_train_policy)
        val_losses.append(val_total_loss)

    return train_losses, val_losses


def evaluate_on_fixed(model, loader, device):
    """在固定评估集上计算加权 value(MSE) / policy(KL) 平均损失。

    权重 = 样本 weight 字段（Σ w·loss / Σ w），与 train_model 训练 loss
    的加权口径一致。2026-09-19 起固定评估集与训练管线对齐（全节点 +
    重分配 policy 标签 + 权重），weight 不再全 1，必须加权——否则 policy
    考分被低权重样本稀释。数据是跨轮次不变的固定评估集，结果可跨轮次
    比较，用于趋势监控、选模型与 LR 调度。
    """
    model.eval()
    total_value = 0.0
    total_policy = 0.0
    total_w = 0.0
    with torch.no_grad():
        for boards, policies, values, weights in loader:
            boards = boards.to(device)
            policies = policies.to(device).view(policies.size(0), -1)
            values = values.to(device).unsqueeze(1)
            w = weights.to(device).unsqueeze(1)      # (B,1)

            pred_values, pred_policies = model(boards)
            vl = (pred_values - values) ** 2
            pl = F.kl_div(F.log_softmax(pred_policies, dim=1),
                          policies, reduction='none').sum(dim=1, keepdim=True)
            total_value += (vl * w).sum().item()
            total_policy += (pl * w).sum().item()
            total_w += w.sum().item()

    if total_w <= 0:
        return 0.0, 0.0
    return total_value / total_w, total_policy / total_w


def make_loader(data, batch_size, shuffle, augment=False):
    boards, policies, values, weights = data
    dataset = WeightedDataset(boards, policies, values, weights,
                              augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _next_run_dir(model_base):
    """扫描 model_base_N 目录，返回下一个可用的路径（如 models/run_3）。"""
    parent = os.path.dirname(model_base) or '.'
    prefix = os.path.basename(model_base)
    if not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
        return os.path.join(parent, f"{prefix}_1")

    existing = []
    for name in os.listdir(parent):
        if name.startswith(f"{prefix}_"):
            try:
                num = int(name.rsplit('_', 1)[1])
                existing.append(num)
            except ValueError:
                continue

    next_num = max(existing, default=0) + 1
    return os.path.join(parent, f"{prefix}_{next_num}")


def _find_prev_champion(model_base, current_run_dir):
    """在 model_base_N 目录中找最近的 champion.pth（排除当前 run）。

    返回路径字符串或 None（首次训练无历史 champion 时）。
    """
    parent = os.path.dirname(model_base) or '.'
    prefix = os.path.basename(model_base)
    if not os.path.isdir(parent):
        return None

    # 规范化路径用于比较
    current_abs = os.path.normpath(os.path.abspath(current_run_dir))

    existing = []
    for name in os.listdir(parent):
        if not name.startswith(f"{prefix}_"):
            continue
        run_dir = os.path.join(parent, name)
        if os.path.normpath(os.path.abspath(run_dir)) == current_abs:
            continue
        champion = os.path.join(run_dir, "champion.pth")
        if os.path.exists(champion):
            try:
                num = int(name.rsplit('_', 1)[1])
            except ValueError:
                continue
            existing.append((num, champion))

    if not existing:
        return None
    existing.sort(key=lambda x: x[0])
    return existing[-1][1]


def work():
    """主训练循环：自对弈 -> 回放池 -> 训练 -> 保存。"""
    config = Config()
    # 自动分配递增的输出目录（models/run_1, run_2, ...）
    run_dir = _next_run_dir(config.model_path)
    os.makedirs(run_dir, exist_ok=True)
    # 固定评估集放在本次 run 目录下，互不干扰
    if config.eval_set_path is None:
        config.eval_set_path = os.path.join(run_dir, 'eval_fixed.pt')

    setup_logging(config.log_path)
    device = get_device()
    # cudnn autotune：输入尺寸固定（batch × 3 × 15 × 15），
    # 与 bf16+channels_last 捆绑启用（基准 C4/C7 实证）
    if torch.cuda.is_available() and getattr(config, "train_amp_bf16", False):
        torch.backends.cudnn.benchmark = True
    logger.info("开始训练，使用设备: %s", device)
    logger.info("输出目录: %s", run_dir)
    if config.sim_ramp_enabled:
        logger.info("配置: 自对弈 %d 局/轮, 模拟次数 %d→%d/步"
                    "（前 %d 轮线性爬升，之后保持上限）, batch=%d, "
                    "epoch=%d, 回放 %d 轮",
                    config.num_samples,
                    config.simulation_start, config.simulation_end,
                    config.sim_ramp_rounds,
                    config.batch_size, config.num_epochs, config.replay_rounds)
    else:
        logger.info("配置: 自对弈 %d 局/轮, 模拟 %d 次/步, batch=%d, "
                    "epoch=%d, 回放 %d 轮, 数据增强=%s, bf16=%s",
                    config.num_samples, config.train_simulation,
                    config.batch_size, config.num_epochs, config.replay_rounds,
                    "全量 D4 8 倍" if config.augment_full else "取样时在线随机 D4",
                    "开" if getattr(config, "train_amp_bf16", False) else "关")

    model = ValueCNN()

    # ---- 基础模型：优先用户指定 base_path，否则继承上一 run 的 champion ----
    if config.base_path is not None:
        model.load_state_dict(
            torch.load(config.base_path, map_location=device, weights_only=True))
        logger.info("已加载基础模型（用户指定）: %s", config.base_path)
    else:
        prev_champion = _find_prev_champion(config.model_path, run_dir)
        if prev_champion is not None:
            model.load_state_dict(
                torch.load(prev_champion, map_location=device, weights_only=True))
            logger.info("已加载基础模型（上一 run champion）: %s", prev_champion)
        else:
            logger.info("无历史 champion，使用全新随机初始化模型")

    model.to(device)

    # ---- 保存基础模型（训练起点，评估集由它生成，可追溯/复用）----
    base_model_path = os.path.join(run_dir, "base_model.pth")
    torch.save(model.state_dict(), base_model_path)
    logger.info("基础模型已保存为 %s", base_model_path)

    # ---- 持久工作进程池：训练全程只创建一次，自对弈 / 竞技场 / 评估集
    #      任务都复用同一批 worker（原实现每轮建池销毁，spawn 开销
    #      ~1.5s/次 × 2000+ 次；worker 内模型也只每轮构建一次）。
    #      worker 为 daemon 进程，主进程异常退出时由系统回收；
    #      正常结束时在训练循环后显式 terminate/join。
    worker_pool = multiprocessing.get_context('spawn').Pool(
        processes=Config.num_workers)
    logger.info("持久工作进程池: 已创建 %d 个 worker（spawn，全程复用）",
                Config.num_workers)

    # ---- 固定评估集：用基础模型生成，跨轮次/跨 run 复用同一份数据 ----
    eval_data = load_fixed_eval_set(config.eval_set_path)
    if eval_data is None:
        logger.info("固定评估集: 未找到缓存 %s，用基础模型生成 %d 局"
                    "（不参与训练）", config.eval_set_path,
                    config.eval_fixed_games)
        model.eval()
        eval_data = generate_fixed_eval_set(model, config.eval_fixed_games,
                                            pool=worker_pool)
        save_fixed_eval_set(config.eval_set_path, eval_data)
        logger.info("固定评估集: 已生成并保存至 %s（增强后 %d 样本）",
                    config.eval_set_path, len(eval_data[0]))
    else:
        logger.info("固定评估集: 已加载缓存 %s（增强后 %d 样本）",
                    config.eval_set_path, len(eval_data[0]))
    eval_fixed_loader = make_loader(eval_data, config.batch_size, shuffle=False)

    # ---- 竞技场卫冕模型：初始为当前模型，新模型胜率达标才晋升 ----
    # champion_step.txt 记录卫冕模型对应的训练轮次，随 champion.pth
    # 一起持久化，保证续训/重启后日志标签仍可追溯。
    os.makedirs(run_dir, exist_ok=True)
    champion_path = os.path.join(run_dir, "champion.pth")
    champion_tag_path = os.path.join(run_dir, "champion_step.txt")
    if os.path.exists(champion_path):
        logger.info("竞技场: 复用已有卫冕模型 %s", champion_path)
        try:
            with open(champion_tag_path, encoding="utf-8") as f:
                champion_step = int(f.read().strip())
        except (OSError, ValueError):
            champion_step = config.train_start - 1
    else:
        torch.save(model.state_dict(), champion_path)
        with open(champion_tag_path, "w", encoding="utf-8") as f:
            f.write(str(config.train_start - 1))
        logger.info("竞技场: 初始卫冕模型已保存为 %s", champion_path)
        champion_step = config.train_start - 1

    # ---- best.pth：近 3 轮 val 总损失滑动平均历史最优的模型 ----
    # 判据用 val（当前模型自对弈的同分布数据，真实反映学习进展）而非
    # 固定评估考卷——考卷锚定 base 模型，模型进步=偏离 base，考卷 loss
    # 与进步脱钩（run_8 实证：val 大幅改善同期考卷 loss 永不更新）。
    # best_val.txt 持久化历史最优值，续训/重启后基线不重置。
    best_path = os.path.join(run_dir, "best.pth")
    best_tag_path = os.path.join(run_dir, "best_val.txt")
    best_val_smooth = None
    if os.path.exists(best_tag_path):
        try:
            best_val_smooth = float(open(best_tag_path,
                                         encoding="utf-8").read().strip())
            logger.info("best 模型: 复用历史最优基线 %.4f（%s）",
                        best_val_smooth, best_path)
        except (OSError, ValueError):
            best_val_smooth = None

    # LR 调度（nn_012 方案）：优化器与 ReduceLROnPlateau 在 train_model
    # 每轮内部新建，不跨轮复用、Adam 动量不跨轮累积。LR 起点恒为
    # config.learning_rate，由当轮验证损失在轮内驱动下调。主循环不再
    # 维护 LR 状态（无 lr_state.txt / arena 反馈）。
    # 近 3 轮 val 总损失滑动平均（best.pth 判据，滤轮间噪声）
    val_total_history = deque(maxlen=3)

    # replay_rounds=0（无回放）时仍需容纳当前轮数据 → maxlen 至少为 1
    replay_buffer = deque(maxlen=max(config.replay_rounds, 1))

    for step in range(config.train_start, config.train_end):
        model.eval()
        logger.info("===== 训练轮次 %d =====", step)

        num_sims = _current_num_simulations(step, config)
        logger.info("自对弈 MCTS 模拟次数: %d", num_sims)
        # 生成原始局数据（未做 D4 增强），回放池只存原始局以节省内存
        games = _generate_games(
            model, config.num_samples, num_simulations=num_sims,
            pool=worker_pool)
        train_games, val_games = split_games(games, Config.train_ratio)
        # 验证集确定性 D4 增强（样本量小，且每轮评估需稳定可比）
        val_data = augment_data(*concat_games(val_games))
        # 回放池只存原始局；D4 增强在 WeightedDataset 取样时在线随机完成
        replay_buffer.append(train_games)
        total_games = sum(len(g) for g in replay_buffer)
        logger.info("回放池: %d/%d 轮，共 %d 局原始数据（%s）",
                    len(replay_buffer), max(config.replay_rounds, 1),
                    total_games,
                    "全量 D4 8 倍增强" if config.augment_full
                    else "取样时在线 D4 增强")

        all_train = _merge_replay(replay_buffer)
        if config.augment_full:
            # 全量增强：物化 D4 全部 8 个视图（训练耗时/内存 ×8）
            all_train = augment_data(*all_train)
        train_loader = make_loader(all_train, config.batch_size, shuffle=True,
                                   augment=not config.augment_full)
        val_loader = make_loader(val_data, config.batch_size, shuffle=False)

        train_losses, val_losses = train_model(
            model, train_loader, val_loader, config)
        logger.info("train losses: %s", [f"{x:.4f}" for x in train_losses])
        logger.info("val losses: %s", [f"{x:.4f}" for x in val_losses])

        # 保存模型（每 arena_interval×2 轮保存一次，频率低于竞技场）
        if (step + 1) % (config.arena_interval * 2) == 0:
            os.makedirs(run_dir, exist_ok=True)
            checkpoint = os.path.join(run_dir, f"{step + 1}.pth")
            torch.save(model.state_dict(), checkpoint)
            logger.info("模型已保存为 %s", checkpoint)

        # ---- 固定评估集：同一份数据上的跨轮次可比 loss ----
        fixed_value, fixed_policy = evaluate_on_fixed(
            model, eval_fixed_loader, device)
        fixed_total = 2 * fixed_value + fixed_policy
        logger.info("[固定评估] value=%.4f policy=%.4f | 总损失=%.4f",
                    fixed_value, fixed_policy, fixed_total)

        # ---- best.pth：近 3 轮 val 总损失滑动平均创新低才覆盖 ----
        val_total_history.append(val_losses[-1])
        val_smooth = sum(val_total_history) / len(val_total_history)
        if best_val_smooth is None or val_smooth < best_val_smooth:
            best_val_smooth = val_smooth
            torch.save(model.state_dict(), best_path)
            with open(best_tag_path, "w", encoding="utf-8") as f:
                f.write(f"{val_smooth:.6f}")
            logger.info("[best] val 滑动平均创新低 %.4f（轮次 %d），已保存 %s",
                        val_smooth, step, best_path)

        # ---- 竞技场：新模型 vs 卫冕模型对弈，胜率达标才晋升 ----
        if (step + 1) % config.arena_interval == 0:
            model.eval()
            stats = pit(model, champion_path,
                        num_games=config.arena_games,
                        simulations=config.arena_simulations,
                        opening_moves=config.arena_opening_moves,
                        pool=worker_pool)
            win_rate = stats["win_rate_a"]
            logger.info(
                "[竞技场] 轮次 %d 对 champion(轮次%d)：%d 局 胜 %d 负 %d 平 %d，"
                "胜率 %.1f%%",
                step, champion_step, stats["num_games"],
                stats["wins_a"], stats["wins_b"], stats["draws"],
                win_rate * 100)
            promoted = (win_rate >= config.arena_promote_rate
                        and stats["wins_a"] > stats["wins_b"])
            if promoted:
                torch.save(model.state_dict(), champion_path)
                with open(champion_tag_path, "w", encoding="utf-8") as f:
                    f.write(str(step))
                champion_step = step
                logger.info("[竞技场] 胜率达到 %.0f%% 阈值，轮次 %d 晋升为"
                            "新卫冕模型",
                            config.arena_promote_rate * 100, step)
            else:
                logger.info("[竞技场] 胜率未达 %.0f%% 阈值，卫冕模型保持不变",
                            config.arena_promote_rate * 100)

    logger.info("训练全部完成（轮次 %d ~ %d）",
                config.train_start, config.train_end - 1)

    # 关闭持久工作进程池（worker 为 daemon，异常退出时也会被运行时回收）
    worker_pool.terminate()
    worker_pool.join()
    logger.info("持久工作进程池已关闭")
    return model


def _merge_replay(replay_buffer):
    """合并回放池各轮的原始局数据（不做 D4 增强）。

    D4 增强默认由 WeightedDataset(augment=True) 在训练取样时在线随机完成，
    避免物化 8 份固定副本导致的 8 倍数据量与重复视图；
    config.augment_full=True 时由调用方在合并后物化全量 8 倍增强。
    """
    all_games = []
    for games in replay_buffer:
        all_games.extend(games)
    return concat_games(all_games)


def _current_num_simulations(step, config):
    """自对弈 MCTS 模拟次数：随轮次从 simulation_start 线性爬升到
    simulation_end，sim_ramp_rounds 轮后保持上限。

    仅用于训练自对弈数据生成；竞技场模拟次数固定，保证胜率跨轮次可比。
    """
    if not getattr(config, "sim_ramp_enabled", False):
        return config.train_simulation
    ramp = max(int(config.sim_ramp_rounds), 1)
    progress = (step - config.train_start) / ramp
    progress = min(max(progress, 0.0), 1.0)
    sims = config.simulation_start + progress * (
        config.simulation_end - config.simulation_start)
    return int(round(sims))


if __name__ == "__main__":
    work()
