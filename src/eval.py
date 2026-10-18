"""模型对局评估：推理封装、模型间对弈（胜率统计）、调试展示。

与训练（train.py / utils.py）和搜索（mcts.py）分离：
本模块只负责“加载模型 -> 对弈 -> 统计胜率 / 展示”。
"""

import copy
import multiprocessing
import random

import numpy as np

from src import board as board_mod
from src.config import Config, board_size
from src.mcts import MCTS, calc_next_move
from src.net import ValueCNN, get_calc, get_device, load_model


class Model:
    """对局推理封装：加载模型并通过 MCTS 选择落子。"""

    def __init__(self, location, simulations=200, c_puct=1.0,
                 dirichlet_alpha=0.0, noise_eps=0.0):
        self.simulations = simulations
        self.model = load_model(location)
        self.mcts = MCTS(self.model, c_puct=c_puct,
                         dirichlet_alpha=dirichlet_alpha, noise_eps=noise_eps)

    def call(self, board, temperature=0, simulations=-1, debug=0):
        if simulations == -1:
            simulations = self.simulations
        value, action_probs = self.mcts.run(copy.deepcopy(board), simulations)
        if debug != 0:
            print("expected_value = ", value)
            print("Probability = ")
            for i in range(board_size):
                for j in range(board_size):
                    if board[i][j] == 1:
                        print("  o", end=" ")
                    elif board[i][j] == 0:
                        print("  x", end=" ")
                    else:
                        print(f"{action_probs[i][j]:.2f}", end=" ")
                print('\n')
        return calc_next_move(board, action_probs, temperature)


def play_game(mcts_a, mcts_b, simulations, a_starts=True, opening_moves=0):
    """两个 MCTS 对弈一局（贪心落子、无探索噪声）。

    opening_moves > 0 时，前若干手随机落子（从空位中均匀采样），
    打破先手必胜的确定性，使胜率出现渐进信号而非 0/50/100%。
    返回 a 视角结果：1 表示 a 胜，-1 表示 b 胜，0 表示平局。
    """
    board = board_mod.new_board()
    order = (mcts_a, mcts_b) if a_starts else (mcts_b, mcts_a)

    for turn in range(board_size * board_size):
        if turn < opening_moves:
            # 随机开局：从空位中均匀采样
            empties = [(i, j) for i in range(board_size)
                       for j in range(board_size) if board[i][j] == 0]
            action = random.choice(empties)
        else:
            mcts = order[turn % 2]
            _value, action_probs = mcts.run(copy.deepcopy(board), simulations)
            action = calc_next_move(board, action_probs, temperature=0)

        board[action[0]][action[1]] = 1
        score = board_mod.evaluation_func(board)
        if score != 0:
            # 胜方为当前落子者
            return 1 if order[turn % 2] is mcts_a else -1
        if board_mod.is_full(board):
            return 0

        board_mod.flip_perspective(board)

    return 0


def _model_spec(model):
    """把模型描述转成可 pickle 的纯数据，供 spawn 子进程重建。

    - None：由调用方处理（表示与对方同一模型）；
    - str：模型 checkpoint 路径，子进程 load_model；
    - nn.Module：取 state_dict + 结构尺寸，子进程重建同构模型。
    """
    if model is None:
        return None
    if isinstance(model, str):
        return {"path": model}
    return {
        # 转 CPU 再传给 spawn 子进程，避免 CUDA tensor IPC 共享在池销毁时
        # 产生 "Producer process terminated" 告警；子进程加载后再 .to(device)。
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "hidden_channels": getattr(model, "hidden_channels", Config.channel),
        "num_blocks": getattr(model, "num_blocks", Config.num_blocks),
    }


def _build_model_from_spec(spec):
    """子进程内按 spec 重建模型并置于评估模式。"""
    if "path" in spec:
        return load_model(spec["path"])
    model = ValueCNN(hidden_channels=spec["hidden_channels"],
                     num_blocks=spec["num_blocks"])
    model.load_state_dict(spec["state"])
    model = model.to(get_device())
    model.eval()
    return model


def _arena_worker(task):
    """子进程：在一批竞技场对局中让两个模型对弈。

    task 为可 pickle 的纯数据（state_dict / 路径 / 局号），spawn 后在
    子进程内重建模型与 MCTS，避免主进程串行对弈的耗时。
    返回 (wins_a, wins_b, draws)。
    """
    (game_indices, spec_a, spec_b, simulations, opening_moves,
     c_puct, infer_batch_size, kwargs_a, kwargs_b) = task

    model_a = _build_model_from_spec(spec_a)
    model_b = (_build_model_from_spec(spec_b) if spec_b is not None
               else model_a)

    kwargs = {}
    if c_puct is not None:
        kwargs["c_puct"] = c_puct
    if infer_batch_size is not None:
        kwargs["infer_batch_size"] = infer_batch_size
    mcts_a = MCTS(model_a, **{**kwargs, **(kwargs_a or {})})
    mcts_b = MCTS(model_b, **{**kwargs, **(kwargs_b or {})})

    wins_a = wins_b = draws = 0
    for g in game_indices:
        result = play_game(mcts_a, mcts_b, simulations,
                           a_starts=(g % 2 == 0),
                           opening_moves=opening_moves)
        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws


def _pit_serial(model_a, model_b, num_games, simulations, c_puct,
                opening_moves, kwargs_a=None, kwargs_b=None):
    """主进程串行对弈（多进程不可用时的回退路径）。"""
    kwargs = {} if c_puct is None else {"c_puct": c_puct}
    mcts_a = MCTS(model_a, **{**kwargs, **(kwargs_a or {})})
    mcts_b = MCTS(model_b, **{**kwargs, **(kwargs_b or {})})

    wins_a = wins_b = draws = 0
    for g in range(num_games):
        result = play_game(mcts_a, mcts_b, simulations,
                           a_starts=(g % 2 == 0),
                           opening_moves=opening_moves)
        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws


def _run_arena_pool(pool, n_workers, model_a, model_b, num_games,
                    simulations, c_puct, opening_moves,
                    kwargs_a=None, kwargs_b=None):
    """把竞技场对局按 worker 切块派发到给定进程池，汇总胜负。"""
    spec_a = _model_spec(model_a)
    spec_b = _model_spec(model_b)
    indices = list(range(num_games))
    # 按局号交错切分，保证每个 worker 的先手局数大致均衡
    chunks = [indices[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]
    tasks = [(chunk, spec_a, spec_b, simulations, opening_moves,
              c_puct, Config.infer_batch_size, kwargs_a, kwargs_b)
             for chunk in chunks]

    wins_a = wins_b = draws = 0
    for wa, wb, dr in pool.imap_unordered(_arena_worker, tasks):
        wins_a += wa
        wins_b += wb
        draws += dr
    return wins_a, wins_b, draws


def pit(model_a, model_b=None, num_games=20, simulations=200, c_puct=None,
        opening_moves=0, num_workers=None, pool=None,
        mcts_kwargs_a=None, mcts_kwargs_b=None):
    """让两个模型对弈 num_games 局（交替先手），统计胜率。

    model_a / model_b 可为模型路径（str）或 nn.Module；
    model_b 为 None 时与 model_a 自身对弈。
    opening_moves > 0 时前若干手随机落子，打破先手必胜确定性。
    mcts_kwargs_a / mcts_kwargs_b：逐方透传给 MCTS 构造器（如
    channels_last / fp16），用于新旧推理路径竞技场验收。
    pool 给定时复用外部持久进程池（与自对弈共用，调用方负责关闭）；
    否则临时建 spawn 池（局数过少/非主进程时回退串行）。
    返回 dict: wins_a / wins_b / draws / win_rate_a。
    """
    if model_b is None:
        model_b = model_a

    if pool is not None:
        n_workers = max(1, min(pool._processes, num_games))
        wins_a, wins_b, draws = _run_arena_pool(
            pool, n_workers, model_a, model_b, num_games,
            simulations, c_puct, opening_moves,
            kwargs_a=mcts_kwargs_a, kwargs_b=mcts_kwargs_b)
        return {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "win_rate_a": wins_a / num_games,
            "num_games": num_games,
        }

    if num_workers is None:
        num_workers = Config.num_workers

    # 多进程条件：主进程、至少 2 个 worker、且局数够分（否则 spawn 开销不划算）
    use_pool = (num_workers >= 2
                and num_games >= 2 * num_workers
                and multiprocessing.current_process().name == 'MainProcess')

    if use_pool:
        with multiprocessing.get_context('spawn').Pool(
                processes=num_workers) as ephemeral_pool:
            wins_a, wins_b, draws = _run_arena_pool(
                ephemeral_pool, num_workers, model_a, model_b, num_games,
                simulations, c_puct, opening_moves,
                kwargs_a=mcts_kwargs_a, kwargs_b=mcts_kwargs_b)
    else:
        wins_a, wins_b, draws = _pit_serial(
            model_a, model_b, num_games, simulations, c_puct, opening_moves,
            kwargs_a=mcts_kwargs_a, kwargs_b=mcts_kwargs_b)

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / num_games,
        "num_games": num_games,
    }


def show_nn(model, board):
    """展示网络对每个空位的先验概率和落子后价值（调试用）。"""
    _, prob = get_calc(model, board)
    val = np.zeros((board_size, board_size), dtype=np.float32)
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] != 0:
                prob[i][j] = 0
            else:
                new_board = copy.deepcopy(board)
                new_board[i][j] = 1
                for x in range(board_size):
                    for y in range(board_size):
                        new_board[x][y] *= -1
                val[i][j], _ = get_calc(model, new_board)
                val[i][j] = float(-val[i][j])
    return prob, val
