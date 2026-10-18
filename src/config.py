"""全局配置。

棋盘约定：board 为 board_size x board_size 的二维列表，
1 表示当前行动方，-1 表示对手，0 表示空位。
"""

board_size = 15


class Config:
    # ---- 训练 ----
    batch_size = 2048          # 训练批大小（2048+bf16 基准测试 3.64x，
                               # 见 test/test_benchmark_train.py；4096 反而变慢）
    train_amp_bf16 = True      # 训练 forward 用 bf16 autocast（loss 用 fp32 还原
                               # 计算，无需 GradScaler）。必须与 channels_last
                               # 捆绑启用：fp32 下 channels_last 是 2 倍负优化，
                               # bf16 下是 1.4 倍正优化（基准 C4/C7/C8 实证）
    num_epochs = 3             # 每轮训练的 epoch 数
    learning_rate = 1e-4       # 每轮初始学习率
    lr_patience = 2            # ReduceLROnPlateau：连续多少 epoch 无改善才降 lr
    lr_factor = 0.5            # ReduceLROnPlateau：触发时 lr × factor
    train_ratio = 0.9          # 按"局"划分训练/验证集的比例
    num_samples = 100          # 每轮自对弈局数
    replay_rounds = 0          # 经验回放：保留最近多少轮的自对弈数据
    augment_full = True        # True=训练集物化全量 D4 8 倍增强（耗时/内存×8）；
                               # False=取样时在线随机单视图
                               # 没学会防守之前建议为 False，学会防守后必须为 True (强AI的关键)。

    # ---- 网络结构 ----
    channel = 32               # 残差网络通道数
    num_blocks = 5             # 残差块数量

    # ---- 自对弈 / MCTS ----
    num_workers = 4            # 自对弈并行 worker 数
    train_simulation = 30      # 每步 MCTS 模拟次数
    sim_ramp_enabled = False   # 模拟次数线性爬升开关
    simulation_start = 100     # 爬升起点（关闭时不生效）
    simulation_end = 600       # 爬升上限
    sim_ramp_rounds = 1200     # 在前多少轮内完成线性爬升
    c_puct = 1.0               # PUCT 探索常数
    dirichlet_alpha = 0.1      # 根节点 Dirichlet 噪声参数
    noise_eps = 0.25           # 根节点噪声混合比例
    opening_max_moves = 20     # 随机开局最大手数（0~20 随机；40 过靠中盘，30 sims 浅树覆盖不到）
    opening_empty_prob = 0.2   # 空盘开局概率

    # ---- 全节点数据收集 ----
    collect_all_nodes = True   # True=搜索树中所有访问过的节点也作为训练数据
    collect_min_frac = 0       # 非根节点收集门槛（visits >= round(sims×frac)）；
                               # 0=全收，软过滤由 weight=sqrt(visits/sims)
                               # 自然降权（低 visits 节点权重 ~0.18）
    policy_reassign = True     # True=prior+Q 前缀重分配，同一分层循环生成
                               # policy 与 value 标签（value 为前缀 max 凸
                               # 组合，nn_012 方案——潜伏威胁价值不被访问量
                               # 稀释）；False=policy 访问分布 + value 树内 Q
    train_buff = 0.8           # 全节点样本权重整体缩放

    # ---- 推理诊断 ----
    eval_profile = False       # 推理耗时分段诊断开关（仅度量用）

    # ---- 计算侧推理优化（只影响推理，训练不动）----
    inference_channels_last = True  # NHWC 内存格式推理
    inference_fp16 = True      # fp16 autocast 推理（启用前需过 test_inference_gate 验收）
    infer_batch_size = 16      # MCTS 叶子批量评估大小（1=逐个推理旧路径；
                               # 30 sims 树浅叶子 ~10-32 个，64 会覆盖整棵
                               # 树导致 Q 反馈进不了访问分布）
    mcts_backend = 'cpp'       # MCTS 树操作后端：'py'=纯 Python，'cpp'=C++ 核心
                               # （需先运行 build_mcts_core.py 编译 src/mcts_core.pyd）
    opening_eval_batch_size = 64  # 随机开局候选局面批量评估大小
    temp_min = 0.1              # 单局温度下限（近贪心）
    temp_max = 0.8              # 单局温度上限（充分探索）

    # ---- 固定评估集（不参与训练）----
    eval_fixed_games = 30      # 固定评估集局数（D4 增强后约 6~8 千样本）

    # ---- 竞技场评估 ----
    arena_interval = 25        # 每多少轮做一次竞技场对弈
    arena_games = 100          # 每次对弈局数（交替先手）
    arena_simulations = 200    # 对弈时每步 MCTS 模拟次数（对弈无探索噪声）
    arena_promote_rate = 0.52  # 胜率达到该值才晋升为新卫冕模型
    arena_opening_moves = 5    # 竞技场随机开局手数，打破先手确定性

    # ---- 路径 ----
    base_path = None           # 基础模型路径（None 则继承上一 run champion）
    model_path = 'models/run'  # 训练输出目录（train.py 自动递增为 run_N）
    eval_set_path = None       # 固定评估集路径（None 则在当前 run 目录下自动生成）
    log_path = 'logging'       # 日志输出目录

    # ---- 训练循环 ----
    train_start = 0            # 起始训练轮次（续训时设为断点轮次）
    train_end = 20000           # 训练结束轮次（不含）
