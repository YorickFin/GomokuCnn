"""棋盘规则测试：落子、翻转、空位、终局（五连）判定。"""

from src import board as B
from src.config import board_size


def test_new_board_empty():
    board = B.new_board()
    assert len(board) == board_size
    assert all(len(row) == board_size for row in board)
    assert B.count_stones(board) == 0
    assert len(B.empties(board)) == board_size * board_size
    assert not B.is_full(board)


def test_flip_perspective():
    board = B.new_board()
    board[0][0] = 1
    board[1][1] = -1
    B.flip_perspective(board)
    assert board[0][0] == -1
    assert board[1][1] == 1
    B.flip_perspective(board)
    assert board[0][0] == 1
    assert board[1][1] == -1


def test_play_move_places_and_flips():
    board = B.new_board()
    board[3][3] = -1  # 对手已有一子
    new_board = B.play_move(board, (7, 7))
    # 原棋盘不被修改
    assert board[7][7] == 0
    # 落子后翻转：新子 1 -> -1，旧子 -1 -> 1
    assert new_board[7][7] == -1
    assert new_board[3][3] == 1


def test_empties_and_count():
    board = B.new_board()
    board[0][0] = 1
    board[0][1] = -1
    assert B.count_stones(board) == 2
    assert len(B.empties(board)) == board_size * board_size - 2
    assert (0, 0) not in B.empties(board)
    assert (0, 1) not in B.empties(board)


# ---------- 终局判定 ----------

def test_winner_horizontal_five():
    board = B.new_board()
    for j in range(5):
        board[7][j] = 1
    assert B.winner(board) == 1
    assert B.evaluation_func(board) > 0


def test_winner_vertical_five_opponent():
    board = B.new_board()
    for i in range(5):
        board[i][3] = -1
    assert B.winner(board) == -1
    assert B.evaluation_func(board) < 0


def test_winner_diagonal_five():
    board = B.new_board()
    for d in range(5):
        board[d][d] = 1
    assert B.winner(board) == 1
    assert B.evaluation_func(board) > 0


def test_winner_anti_diagonal_five():
    board = B.new_board()
    for d in range(5):
        board[d][10 - d] = -1
    assert B.winner(board) == -1
    assert B.evaluation_func(board) < 0


def test_no_five():
    board = B.new_board()
    for j in range(4):
        board[0][j] = 1
    assert B.winner(board) == 0
    assert B.evaluation_func(board) == 0


def test_six_in_a_row_still_wins():
    """六连中包含五连，应判胜。"""
    board = B.new_board()
    for j in range(6):
        board[7][j] = 1
    assert B.winner(board) == 1


def test_is_full():
    board = B.new_board()
    assert not B.is_full(board)
    for i in range(board_size):
        for j in range(board_size):
            board[i][j] = 1 if (i + j) % 2 == 0 else -1
    assert B.is_full(board)
