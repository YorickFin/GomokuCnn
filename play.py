"""五子棋人机对弈 —— 桌面后端（pywebview 桥）。

前端代码位于 frontend/（Vue 3 + Vite），两种运行模式：

  * 发布模式（默认）：直接 python play.py。pywebview 会启动内置
    HTTP 服务器加载 frontend/dist 构建产物，无需 Node 环境；
    修改前端代码后需在 frontend 目录重新执行 pnpm build。
  * 开发模式：把下方 FRONTEND_URL 改为 'http://localhost:5173/'，
    先在 frontend 目录执行 pnpm dev，再运行 python play.py，
    可享受 Vite HMR 热更新。

pywebview 会把 js_api 桥注入窗口内页面，前端通过
window.pywebview.api 调用本文件中的对局接口，无需独立 HTTP 后端。

棋盘固定视角约定：黑棋 = 1，白棋 = -1，空位 = 0；
送入 MCTS 前按当前行动方翻转为“己方=1”的视角。
"""

import os
import sys
import threading
import time
import urllib.request

# private_mode=False 时 pywebview 用固定端口 42001，且 WebView2 会把该
# origin 的 HTTP 缓存持久化到 storage_path 目录（EBWebView/Default/Cache），
# 前端 dist 更新后窗口仍可能加载旧缓存资源。通过 WebView2 标准环境变量
# 禁用 HTTP 缓存（在 webview.start 创建 WebView2 环境前设置即可生效）。
os.environ['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = '--disable-http-cache'

import webview

from src import board as board_mod
from src.config import board_size
from src.mcts import MCTS, calc_next_move
from src.net import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# play.py 不参与训练，直接指向 models 父目录以遍历所有 run_N 子目录。
MODEL_DIR = os.path.join(BASE_DIR, 'models')
# 发布模式：加载 Vite 构建产物（pywebview 内置 HTTP 服务器提供服务）。
# 开发模式：改为 'http://localhost:5173/' 并先启动 pnpm dev。
FRONTEND_URL = os.path.join(BASE_DIR, 'frontend', 'dist', 'index.html')
DEFAULT_SIMULATIONS = 200  # AI 每步 MCTS 模拟次数默认值
SIMULATION_CHOICES = [200, 600, 1200, 1800, 2400, 3600]   # UI 可选强度档位


class GomokuAPI:
    """pywebview 暴露给前端的接口（所有方法均在 API 线程被调用）。"""

    def __init__(self):
        self.model = None
        self.mcts = None
        self.heat_mcts = None          # 热力图专用（与对局树互不污染）
        self._tree_root = None         # 跨手树复用：当前搜索树根 id
        self.model_name = None
        self.temperature = 0.0         # 温度：0.0 为确定性，1.0 为随机性
        self.simulations = DEFAULT_SIMULATIONS   # 当前 MCTS 模拟次数（UI 可调）
        self.edit_mode = False         # 编辑模式：自由摆子，不触发 AI
        self.lock = threading.Lock()
        self._reset_game(human_black=True)

    def set_simulations(self, n):
        """设置 AI 每步 MCTS 模拟次数（对局中可随时调整，下一手生效）。"""
        n = int(n)
        if n not in SIMULATION_CHOICES:
            return {'error': '不支持的模拟次数：%s' % n}
        with self.lock:
            self.simulations = n
        return {'ok': True, 'simulations': n}

    # ---- 模型管理 ----
    def list_models(self):
        """返回 models/run_N/ 下全部 checkpoint 的相对路径，按
        (run 号, 文件类型, 步数) 升序排列。

        返回形如 ['run_1/1.pth', ..., 'run_1/champion.pth',
                  'run_2/1.pth', ..., 'run_2/champion.pth']。
        前端选最后一项即为最新 run 的 champion.pth。
        """
        if not os.path.isdir(MODEL_DIR):
            return []
        runs = []
        for name in os.listdir(MODEL_DIR):
            full = os.path.join(MODEL_DIR, name)
            if os.path.isdir(full) and name.startswith('run_'):
                try:
                    run_no = int(name.split('_', 1)[1])
                except (ValueError, IndexError):
                    run_no = 0
                runs.append((run_no, name, full))
        runs.sort(key=lambda x: x[0])

        def file_key(fname):
            stem = os.path.splitext(fname)[0]
            if stem == 'base_model':
                return (0, 0)        # run 内最前
            if stem == 'champion':
                return (2, 0)        # run 内最后
            if stem.isdigit():
                return (1, int(stem))
            return (1, 0)

        out = []
        for _, run_name, full in runs:
            pths = sorted(
                (f for f in os.listdir(full) if f.endswith('.pth')),
                key=file_key,
            )
            for f in pths:
                # 用 '/' 统一相对路径分隔符，便于前端展示与传回
                out.append(f"{run_name}/{f}")
        return out

    def load_model(self, name):
        """加载指定 checkpoint（切换模型后自动新开一局由前端触发）。"""
        path = os.path.join(MODEL_DIR, name)
        model = load_model(path)
        with self.lock:
            self.model = model
            self.mcts = MCTS(model)
            self.heat_mcts = MCTS(model)   # 独立实例，避免热力图污染对局树
            self._tree_root = None
            self.model_name = name
        return {'ok': True, 'name': name}

    # ---- 对局流程 ----
    def set_edit_mode(self, on):
        """切换编辑模式。

        开启：清空终局状态、丢弃搜索树，用户可自由交替黑白落子摆局面，
        不触发 AI、不判胜负。
        关闭：重新判定终局；随后重新分配黑白方——轮到的一方归 AI
        （AI 必定先落子），玩家执另一方，之后恢复"玩家落子 → AI 应手"
        的正常对局。
        """
        with self.lock:
            if self.mcts is None:
                return {'error': '模型未加载'}
            on = bool(on)
            if on == self.edit_mode:
                return self._state()
            self.edit_mode = on
            self._tree_root = None     # 局面将被/已被手工修改，搜索树作废
            if on:
                # 允许在已终局（over）的盘面上重新摆子
                self.over = False
                self.winner = None
                self.win_line = []
            else:
                self._recheck_terminal()
                if not self.over:
                    # 重新分配黑白方：轮到的一方归 AI，玩家执另一方
                    self.human_black = (self.move_count % 2 == 1)
                    self._ai_move()
            return self._state()

    def clear_board(self):
        """编辑模式专用：清空棋盘从空盘开始摆（不退出编辑模式）。"""
        with self.lock:
            if self.mcts is None:
                return {'error': '模型未加载'}
            if not self.edit_mode:
                return {'error': '仅在编辑模式下可清空棋盘'}
            board = board_mod.new_board()
            self.board = board
            self.last_move = None
            self.win_line = []
            self.move_count = 0
            self.over = False
            self.winner = None
            self._tree_root = None
            return self._state()

    def start_game(self, human_black):
        """开始新对局；AI 先手时直接走完 AI 的第一手再返回。"""
        with self.lock:
            if self.mcts is None:
                return {'error': '模型未加载'}
            self.edit_mode = False     # 新对局自动退出编辑模式
            self._reset_game(human_black=bool(human_black))
            if not self.human_black:      # AI 执黑先手
                self._ai_move()
            return self._state()

    def human_move(self, r, c):
        """玩家落子；若未终局则紧接着走 AI 的一手。

        编辑模式下：按手数黑白交替落子，不触发 AI、不判胜负。
        """
        with self.lock:
            r, c = int(r), int(c)
            if self.mcts is None:
                return {'error': '模型未加载'}
            if self.edit_mode:
                if self.over or not board_mod.in_bounds(r, c) \
                        or self.board[r][c] != 0:
                    return self._state()
                color = 1 if self.move_count % 2 == 0 else -1   # 黑先白后
                self._edit_place(r, c, color)
                return self._state()
            if self.over or not self._is_human_turn():
                return self._state()
            if not board_mod.in_bounds(r, c) or self.board[r][c] != 0:
                return self._state()
            self._place(r, c, 1 if self.human_black else -1)
            if not self.over:
                # 玩家落子通常在 AI 上一手搜索树的 children 里，
                # reroot 过去供 AI 本手继承搜索积累（跨手树复用）
                if self._tree_root is not None:
                    self._tree_root = self.mcts.next_root(
                        self._tree_root, (r, c))
                self._ai_move()
            return self._state()

    def get_heatmap(self):
        """概率热力图：对当前局面做一次 MCTS，返回根节点访问概率分布。

        仅在轮到玩家且对局未结束时有意义。probs 为 15x15 矩阵
        （坐标与棋盘一致），value 已统一换算为“玩家视角”局势价值
        （[-1,1]，越大对玩家越有利），前端无需再做视角转换。
        """
        with self.lock:
            if self.mcts is None:
                return {'error': '模型未加载'}
            if self.edit_mode or self.over or not self._is_human_turn():
                return {'probs': None, 'value': None}
            if board_mod.count_stones(self.board) == 0:
                return {'probs': None, 'value': None}
            # 翻转为“当前行动方（=玩家）视角”
            persp = board_mod.clone_board(self.board)
            black_to_move = (self.move_count % 2 == 0)
            if not black_to_move:
                board_mod.flip_perspective(persp)
            # 热力图用独立 MCTS 实例：玩家视角的搜索树与 AI 对局树
            # （AI 视角 + 跨手复用）互不污染
            value, probs = self.heat_mcts.run(persp, self.simulations)
            return {
                'probs': [[float(x) for x in row] for row in probs],
                'value': float(value),
            }

    # ---- 内部逻辑 ----
    def _reset_game(self, human_black):
        self.board = board_mod.new_board()   # 固定视角：黑=1，白=-1
        self.human_black = bool(human_black)
        self.over = False
        self.winner = None                   # 'human' / 'ai' / 'draw'
        self.last_move = None
        self.win_line = []
        self.move_count = 0
        self._tree_root = None               # 新开局丢弃旧搜索树

    def _is_human_turn(self):
        black_to_move = (self.move_count % 2 == 0)
        return black_to_move == self.human_black

    def _ai_move(self):
        ai_color = 1 if not self.human_black else -1
        # MCTS 需要“当前行动方视角”：AI 执白时翻转棋盘
        persp = board_mod.clone_board(self.board)
        if ai_color == -1:
            board_mod.flip_perspective(persp)
        # 自对弈训练多从随机中盘开局，空盘先验覆盖不足，
        # MCTS 会选出边角怪招；空盘按定式直接走天元。
        if board_mod.count_stones(persp) == 0:
            r = c = board_size // 2
            self._tree_root = None   # 天元未经搜索，无从 reroot，下手从零
        else:
            # 跨手树复用：从上一手继承的树根继续搜（每手固定 sims 数，
            # 但继承子树的访问统计/Q 直接复用，等效强度更高）
            (_value, probs), new_root = self.mcts.run(
                persp, self.simulations,
                cur_root=self._tree_root, return_root=True)
            r, c = calc_next_move(persp, probs, temperature=self.temperature)
            # reroot 到 AI 实际落点；落点未建 child 时返回 None（下手从零）
            self._tree_root = self.mcts.next_root(new_root, (int(r), int(c)))
        self._place(int(r), int(c), ai_color)

    def _edit_place(self, r, c, color):
        """编辑模式落子：只改棋盘与手数，不判终局、不触发 AI。"""
        self.board[r][c] = color
        self.last_move = [r, c]
        self.move_count += 1

    def _recheck_terminal(self):
        """编辑模式结束后重新判定终局：全盘扫描五连与满盘。"""
        self.over = False
        self.winner = None
        self.win_line = []
        for r in range(board_size):
            for c in range(board_size):
                color = self.board[r][c]
                if color == 0:
                    continue
                line = self._five_line(r, c, color)
                if line is not None:
                    self.over = True
                    self.win_line = [list(p) for p in line]
                    human_color = 1 if self.human_black else -1
                    self.winner = 'human' if color == human_color else 'ai'
                    return
        if board_mod.is_full(self.board):
            self.over = True
            self.winner = 'draw'

    def _place(self, r, c, color):
        self.board[r][c] = color
        self.last_move = [r, c]
        self.move_count += 1
        line = self._five_line(r, c, color)
        if line is not None:
            self.over = True
            self.win_line = [list(p) for p in line]
            human_color = 1 if self.human_black else -1
            self.winner = 'human' if color == human_color else 'ai'
        elif board_mod.is_full(self.board):
            self.over = True
            self.winner = 'draw'

    def _five_line(self, r, c, color):
        """返回刚落子形成的五连同色坐标列表（无五连返回 None）。"""
        for dr, dc in board_mod.DIRECTIONS:
            line = [(r, c)]
            rr, cc = r + dr, c + dc
            while board_mod.in_bounds(rr, cc) and self.board[rr][cc] == color:
                line.append((rr, cc))
                rr, cc = rr + dr, cc + dc
            rr, cc = r - dr, c - dc
            while board_mod.in_bounds(rr, cc) and self.board[rr][cc] == color:
                line.insert(0, (rr, cc))
                rr, cc = rr - dr, cc - dc
            if len(line) >= 5:
                return line
        return None

    def _state(self):
        return {
            'board': self.board,
            'over': self.over,
            'winner': self.winner,
            'lastMove': self.last_move,
            'winLine': self.win_line,
            'moveCount': self.move_count,
            'humanBlack': self.human_black,
            'humanTurn': (not self.over) and not self.edit_mode \
                and self._is_human_turn(),
            'editMode': self.edit_mode,
            'simulations': self.simulations,
            'model': self.model_name,
        }


def _wait_for_dev_server(url, timeout=20):
    """开发模式：等待 Vite 开发服务器就绪，避免窗口先打开而显示连接失败页。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except Exception:
            print('等待前端开发服务器 %s …（请先在 frontend 目录运行 pnpm dev）'
                  % url)
            time.sleep(1)
    print('未连接到前端开发服务器 %s，仍将打开窗口。' % url)
    return False


def _check_frontend_ready(url):
    """启动前检查前端入口可用。

    http(s) 地址：轮询等待开发服务器；
    本地文件路径：检查构建产物是否存在（pywebview 会用内置
    HTTP 服务器服务该文件所在目录）。
    """
    if url.startswith(('http://', 'https://')):
        return _wait_for_dev_server(url)
    if not os.path.isfile(url):
        print('未找到前端构建产物：%s' % url)
        print('请先在 frontend 目录执行：pnpm build')
        return False
    return True


def main():
    if not _check_frontend_ready(FRONTEND_URL):
        sys.exit(1)
    api = GomokuAPI()
    webview.create_window(
        '五子棋 AI 对弈',
        url=FRONTEND_URL,
        js_api=api,
        width=980,
        height=700,
        min_size=(900, 640),
        background_color='#f2efe8',
    )
    # private_mode=True（默认）时 pywebview 用
    # tempfile.TemporaryDirectory().name 建 Edge 用户数据目录，该目录
    # 会随临时对象被 GC 立即删除，导致 WebView2 初始化失败（0x8000FFFF，
    # "无法读取和写入其数据目录"）。关闭隐私模式改用固定目录 %APPDATA%\pywebview。
    webview.start(
        private_mode=False,
        storage_path=os.path.join(os.environ['LOCALAPPDATA'], 'GomokuCnn', 'webview'),
    )


if __name__ == '__main__':
    main()
