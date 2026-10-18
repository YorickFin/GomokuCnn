/** 对局状态（与 play.py 的 GomokuAPI._state() 对应）。 */
export interface GameState {
  /** 15x15 棋盘：1=黑棋，-1=白棋，0=空位 */
  board: number[][]
  over: boolean
  /** 'human' 玩家胜 / 'ai' AI 胜 / 'draw' 平局 / null 未结束 */
  winner: 'human' | 'ai' | 'draw' | null
  lastMove: [number, number] | null
  winLine: [number, number][]
  moveCount: number
  /** 玩家是否执黑（先手） */
  humanBlack: boolean
  /** 当前是否轮到玩家（编辑模式下恒为 false） */
  humanTurn: boolean
  /** 是否处于编辑模式（自由摆子，不触发 AI） */
  editMode: boolean
  simulations: number
  model: string | null
}

/** 概率热力图接口返回。 */
export interface HeatmapResult {
  /** 15x15 MCTS 根节点访问概率；不可用/未开启时为 null */
  probs: number[][] | null
  /** 玩家视角局势价值 [-1,1]，越大对玩家越有利 */
  value: number | null
  error?: string
}

/** 带错误信息的对局状态返回。 */
export type StateResult = GameState & { error?: string }
