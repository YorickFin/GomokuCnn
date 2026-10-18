"""棋盘规则：落子、视角翻转、终局（五连）判定。

棋盘约定：board 为 board_size x board_size 的二维列表，
1 表示当前行动方，-1 表示对手，0 表示空位。
本模块只包含无副作用的纯规则逻辑，不依赖网络与搜索。
"""

import copy

from src.config import board_size

# 横、竖、主对角、副对角
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def new_board():
    """返回空棋盘。"""
    return [[0 for _ in range(board_size)] for _ in range(board_size)]


def clone_board(board):
    """深拷贝棋盘。"""
    return copy.deepcopy(board)


def count_stones(board):
    """已落子总数。"""
    return sum(1 for row in board for v in row if v != 0)


def empties(board):
    """所有空位坐标列表。"""
    return [(i, j)
            for i in range(board_size) for j in range(board_size)
            if board[i][j] == 0]


def is_full(board):
    """棋盘是否下满。"""
    for row in board:
        for v in row:
            if v == 0:
                return False
    return True


def in_bounds(i, j):
    return 0 <= i < board_size and 0 <= j < board_size


def flip_perspective(board):
    """原地翻转视角：1 与 -1 互换（落子后交由对手行动）。"""
    for i in range(board_size):
        for j in range(board_size):
            board[i][j] *= -1
    return board


def play_move(board, move):
    """返回新棋盘：当前方（1）在 move 落子，然后翻转视角。

    落子与翻转在同一函数内完成，与 MCTS 中“落子后交换行动方”
    的约定保持一致。
    """
    i, j = move
    new_board_ = clone_board(board)
    new_board_[i][j] = 1
    flip_perspective(new_board_)
    return new_board_


def winner(board):
    """返回胜方：1 表示当前方五连，-1 表示对手五连，0 表示无五连。"""
    for i in range(board_size):
        for j in range(board_size):
            v = board[i][j]
            if v == 0:
                continue
            for dx, dy in DIRECTIONS:
                cnt = 0
                for d in range(5):
                    ni, nj = i + d * dx, j + d * dy
                    if in_bounds(ni, nj) and board[ni][nj] == v:
                        cnt += 1
                    else:
                        break
                if cnt == 5:
                    return v
    return 0


def evaluation_func(board):
    """终局评估分。

    1 五连返回正分、-1 五连返回负分，分值随落子数衰减
    （鼓励速胜速负识别）；无五连返回 0。
    """
    w = winner(board)
    if w == 0:
        return 0
    return w * (1 - count_stones(board) * 3e-4)
