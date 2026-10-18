"""量化随机开局候选的多样性。

问题：generate_random_board 中 perm 只 shuffle 一次，
所有 num_run 个候选都取同一个 perm 的前缀，只是长度不同。
这导致候选之间高度相关，选最均势的候选时几乎没有真正的选择空间。

本脚本复制候选生成逻辑（不含模型评估），统计：
1. 完全重复候选比例
2. 候选两两平均重叠度（相同位置同色棋子比例）
3. 唯一开局比例
4. 不同长度候选的分布
"""

import random

import numpy as np

from src import board as board_mod
from src.config import Config, board_size


def generate_candidates_only(seed=None):
    """复制 generate_random_board 的候选生成部分，不做模型评估。

    返回候选棋盘列表。
    """
    if seed is not None:
        random.seed(seed)

    all_positions = [(i, j) for i in range(board_size)
                     for j in range(board_size)]

    if random.random() < Config.opening_empty_prob:
        num_run = 0
    else:
        num_run = random.randint(200, 1000)

    candidates = []
    move_counts = []
    for _ in range(num_run):
        num_moves = random.randint(0, Config.opening_max_moves)
        board = board_mod.new_board()
        current_player = 1
        for i, j in random.sample(all_positions, num_moves):
            board[i][j] = current_player
            current_player = -current_player
        if board_mod.evaluation_func(board) != 0:
            continue
        if current_player == -1:
            board_mod.flip_perspective(board)
        candidates.append(board)
        move_counts.append(num_moves)

    return candidates, move_counts, num_run


def board_to_hashable(board):
    """把棋盘转成可哈希的 tuple，用于去重。"""
    return tuple(tuple(row) for row in board)


def overlap(b1, b2):
    """计算两个棋盘的重叠度：相同位置上有相同颜色棋子的比例。

    只统计至少一方有子的位置。
    """
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)
    both_nonempty = (b1 != 0) | (b2 != 0)
    if not both_nonempty.any():
        return 1.0
    same = (b1 == b2) & both_nonempty
    return float(same.sum()) / float(both_nonempty.sum())


def jaccard_positions(b1, b2):
    """Jaccard 相似度：双方有子位置的交集 / 并集。"""
    s1 = set((i, j) for i in range(board_size) for j in range(board_size)
             if b1[i][j] != 0)
    s2 = set((i, j) for i in range(board_size) for j in range(board_size)
             if b2[i][j] != 0)
    if not s1 and not s2:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 1.0


def analyze_candidates(candidates, move_counts, num_run, label=""):
    print(f"\n{'='*60}")
    print(f"分析 {label}")
    print(f"{'='*60}")
    print(f"num_run (尝试次数): {num_run}")
    print(f"有效候选数 (跳过终局): {len(candidates)}")

    if not candidates:
        print("无有效候选。")
        return

    # 1. 完全重复
    hashes = [board_to_hashable(b) for b in candidates]
    unique_hashes = set(hashes)
    n_unique = len(unique_hashes)
    dup_rate = 1.0 - n_unique / len(candidates)
    print(f"\n[1] 完全重复:")
    print(f"  唯一候选数: {n_unique} / {len(candidates)}")
    print(f"  重复率: {dup_rate:.2%}")

    # 2. 长度分布
    from collections import Counter
    mc = Counter(move_counts)
    print(f"\n[2] 候选落子数分布 (前 10 个):")
    for k, v in sorted(mc.items())[:10]:
        print(f"  {k} 手: {v} 个")
    print(f"  ... 共 {len(mc)} 种长度")

    # 3. 两两重叠度（采样，避免 O(n^2) 过大）
    n = len(candidates)
    sample_pairs = 2000
    overlaps = []
    jaccards = []
    for _ in range(min(sample_pairs, n * (n - 1) // 2)):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        overlaps.append(overlap(candidates[i], candidates[j]))
        jaccards.append(jaccard_positions(candidates[i], candidates[j]))

    print(f"\n[3] 两两相似度 (采样 {len(overlaps)} 对):")
    print(f"  同色重叠度 均值={np.mean(overlaps):.3f} "
          f"中位数={np.median(overlaps):.3f}")
    print(f"  位置 Jaccard 均值={np.mean(jaccards):.3f} "
          f"中位数={np.median(jaccards):.3f}")

    # 4. 前缀关系检测：有多少候选是另一个候选的严格前缀
    #    （即 A 的所有落子位置在 B 中同色存在，且 B 更长）
    prefix_pairs = 0
    checked = 0
    for _ in range(min(sample_pairs, n * (n - 1) // 2)):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j or move_counts[i] >= move_counts[j]:
            continue
        checked += 1
        bi = candidates[i]
        bj = candidates[j]
        is_prefix = True
        for r in range(board_size):
            for c in range(board_size):
                if bi[r][c] != 0 and bi[r][c] != bj[r][c]:
                    is_prefix = False
                    break
            if not is_prefix:
                break
        if is_prefix:
            prefix_pairs += 1

    print(f"\n[4] 前缀关系 (短候选是长候选的前缀):")
    if checked > 0:
        print(f"  前缀对比例: {prefix_pairs}/{checked} "
              f"= {prefix_pairs/checked:.2%}")
    else:
        print(f"  无可比较的长短对")

    return dup_rate, np.mean(jaccards)


def main():
    random.seed(42)
    np.random.seed(42)

    print(f"Config.opening_empty_prob = {Config.opening_empty_prob}")
    print(f"Config.opening_max_moves = {Config.opening_max_moves}")
    print(f"Config.opening_eval_batch_size = {Config.opening_eval_batch_size}")

    # 分析多次调用，每次 perm 不同
    all_dup = []
    all_jacc = []
    for trial in range(8):
        cands, mcs, nr = generate_candidates_only(seed=100 + trial)
        if not cands:
            print(f"\n第 {trial+1} 次调用: num_run=0（空盘开局，跳过分析）")
            continue
        dup, jacc = analyze_candidates(cands, mcs, nr,
                                       label=f"第 {trial+1} 次调用")
        all_dup.append(dup)
        all_jacc.append(jacc)

    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"平均重复率: {np.mean(all_dup):.2%}")
    print(f"平均位置 Jaccard: {np.mean(all_jacc):.3f}")
    print(f"\n结论参考：")
    print(f"- 若重复率高 & Jaccard 高，说明候选几乎同质，")
    print(f"  选最均势候选时没有真正的多样性。")
    print(f"- 根因：所有候选共享同一个 perm，只是前缀长度不同。")


if __name__ == "__main__":
    main()
