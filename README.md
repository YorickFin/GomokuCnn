# GomokuCnn

基于 CNN + MCTS 自对弈强化学习的五子棋 AI。采用低算力（30 sims）训练管线，在消费级显卡（RTX 3060 Laptop）上从零训出强 AI，并附带一套桌面人机对弈程序。

本项目参考了 [zhoukangyang233/Gomoku](https://github.com/zhoukangyang233/Gomoku) 的训练方案（nn_012 管线），并在此基础上做了大量工程优化与机制研究。

## 特性

- **低 sims 自对弈训练管线**：30 次 MCTS 模拟/步 + 全节点数据收集 + prior+Q 分层标签重分配，用总轮次换取棋力（详见 docs）
- **C++ MCTS 核心**：pybind11 加速树操作（33.9x），批量叶子推理，fp16 + channels_last 推理优化
- **物化 D4 八倍增强**：训练集 8 方位物化对称，破除自对弈闭环的方位偏置
- **value 标签前缀重分配**：policy 与 value 标签由同一分层循环产出，value 头学会嗅探潜伏威胁
- **桌面人机对弈**：pywebview + Vue 3 前端，支持强度档位、热力图、编辑模式、MCTS 跨手树复用
- **竞技场比赛与日志分析**：arena 晋升机制、gauntlet 模型海选、训练日志分析工具

## 棋力表现

当前最强模型 [models/run_6/950.pth](models/run_6/950.pth) 与传统 NNUE 引擎 [rapfi](https://github.com/dhbloo/rapfi) 的对抗实测：

- **先手执黑**：基本能做到 100% 胜率；
- **后手执白与平衡开局**：表现不佳。

这两类局面的补强需要另做专项训练（后手防守、平衡开局），不在当前自对弈管线的覆盖范围内，属于后续方向。

## 项目结构

```
├── train.py            # 自对弈强化学习训练入口
├── play.py             # 人机对弈桌面程序（pywebview 桥）
├── analyze_log.py      # 训练日志分析
├── build_mcts_core.py  # 编译 C++ MCTS 核心（生成 src/mcts_core.pyd）
├── src/                # 核心库：棋盘 / 网络 / MCTS / 配置 / 评估
│   ├── mcts_core.cpp   # C++ MCTS 树实现
│   └── ...
├── frontend/           # Vue 3 + Vite 人机对弈前端
├── test/               # pytest 测试与基准
├── models/             # 训练产出的模型（run_N 目录）
└── docs/               # 机制研究文档
```

## 快速开始

### 环境准备

```bash
# Python 依赖（uv，Python 3.13）
uv sync

# 编译 C++ MCTS 核心（需要 MinGW-w64 / UCRT64 g++）
python build_mcts_core.py

# 前端依赖（pnpm，仅修改前端时需要）
cd frontend
pnpm install
pnpm build
```

### 训练

```bash
python train.py
```

每轮流程：自对弈生成数据（4 worker 并行）→ 物化 D4 增强 → 训练（每轮重建 Adam + ReduceLROnPlateau 轮内调度）→ 定期 arena 对战评估晋升。

### 人机对弈

```bash
python play.py
```

加载 `frontend/dist` 构建产物，无需 Node 环境。开发模式可将 `play.py` 中 `FRONTEND_URL` 改为 `http://localhost:5173/` 并在 frontend 目录执行 `pnpm dev` 以获得热更新。

## 文档

本项目在调参过程中沉淀了三篇机制研究文档，记录了低算力自对弈强化学习的关键结论：

- [低 sims 下棋类游戏强化学习如何逃出双盲期](docs/低%20sims%20下棋类游戏强化学习如何逃出双盲期.md) —— 冷启动"双盲期"的定义、成因、诊断指标与逃逸手段：为什么模型初期不会防守、胜率暴涨为何是假象，以及 30 sims 管线如何靠标签机制与总轮次打破低水平均衡。
- [自对弈强化学习成为强 AI 的硬约束——物化 D4 对称增强的不可替代性](docs/自对弈强化学习成为强%20AI%20的硬约束——物化%20D4%20对称增强的不可替代性.md) —— 自对弈闭环的方位偏置是正反馈，物化 D4 八倍增强是信息论意义上的硬约束，Online D4 + 经验回放无法替代。
- [自对弈强化学习的潜伏威胁盲区——value 标签前缀重分配](docs/自对弈强化学习的潜伏威胁盲区——value%20标签前缀重分配.md) —— 防杀能力不在搜索深度而在训练标签：value 标签必须与 policy 标签在同一分层循环中以前缀 max 凸组合产出，否则潜伏杀点信号被访问量稀释，模型对强算杀对手结构性失守。

## 参考与致谢

- [zhoukangyang233/Gomoku](https://github.com/zhoukangyang233/Gomoku) —— 本项目训练方案（30 sims 低算力管线、全节点收集、分层标签重分配、每轮重建优化器的 LR 方案等）以该项目（nn_012 管线）为参考实现，谨致谢意。

## License

本项目以 [MIT License](LICENSE) 开源。
