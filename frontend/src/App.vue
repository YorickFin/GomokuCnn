<script setup lang="ts">
/**
 * 五子棋人机对弈主界面：模型选择、新游戏、先后手切换、
 * 概率热力图开关与局势胜率条。对局接口经 pywebview 桥调用 play.py。
 */
import { computed, onMounted, ref } from 'vue'
import GomokuBoard from './components/GomokuBoard.vue'
import { waitForBridge } from './bridge'
import type { GameState, HeatmapResult } from './types'

type PywebviewApi = NonNullable<Awaited<ReturnType<typeof waitForBridge>>>

const api = ref<PywebviewApi | null>(null)
const bridgeMissing = ref(false)

const SIM_CHOICES = [200, 600, 1200, 1800, 2400, 3600]

const models = ref<string[]>([])
const selectedModel = ref('')
const state = ref<GameState | null>(null)
const busy = ref(false)
const humanBlack = ref(true)
const selectedSim = ref(200)
const heatOn = ref(false)
const heatProbs = ref<number[][] | null>(null)
const heatValue = ref<number | null>(null)
const notice = ref('准备中…')
const noticeSub = ref('正在初始化')
let heatToken = 0

// ---------- 状态展示 ----------
const editMode = computed(() => state.value?.editMode ?? false)

// 编辑模式下悬停预览的颜色 = 当前手数应落色（黑先白后）
const hoverBlack = computed(() => {
  if (editMode.value) {
    return (state.value?.moveCount ?? 0) % 2 === 0
  }
  return humanBlack.value
})

const turnDotClass = computed(() => {
  if (editMode.value) {
    return (state.value?.moveCount ?? 0) % 2 === 0 ? 'black' : 'white'
  }
  if (state.value?.over) {
    if (state.value.winner === 'draw') return 'black'
    const humanWon = state.value.winner === 'human'
    return humanWon === humanBlack.value ? 'black' : 'white'
  }
  if (state.value && !state.value.humanTurn) {
    return humanBlack.value ? 'white' : 'black'
  }
  return humanBlack.value ? 'black' : 'white'
})

const winPct = computed(() => {
  if (heatValue.value === null) return null
  return Math.max(2, Math.min(98, Math.round(((heatValue.value + 1) / 2) * 100)))
})

function renderStatus() {
  if (bridgeMissing.value) {
    notice.value = '未检测到桌面桥接'
    noticeSub.value = '请运行 python play.py 从桌面窗口打开'
    return
  }
  const s = state.value
  if (!s) return
  const colorName = humanBlack.value ? '黑棋' : '白棋'
  if (s.editMode) {
    notice.value = '编辑模式'
    noticeSub.value =
      `第 ${s.moveCount + 1} 手 · 轮到${s.moveCount % 2 === 0 ? '黑' : '白'}方落子`
    return
  }
  if (s.over) {
    if (s.winner === 'human') {
      notice.value = '你赢了！'
      noticeSub.value = `共 ${s.moveCount} 手`
    } else if (s.winner === 'ai') {
      notice.value = 'AI 获胜'
      noticeSub.value = `共 ${s.moveCount} 手 · 再来一局？`
    } else {
      notice.value = '平局'
      noticeSub.value = '棋盘下满'
    }
  } else if (s.humanTurn && !busy.value) {
    notice.value = '轮到你落子'
    noticeSub.value = `你执${colorName} · 第 ${s.moveCount + 1} 手`
  } else {
    notice.value = 'AI 思考中…'
    noticeSub.value = `MCTS ${s.simulations} 次模拟`
  }
}

// ---------- 概率热力图 ----------
async function refreshHeatmap() {
  const s = state.value
  const canShow = heatOn.value && s && !s.over && s.humanTurn && !busy.value
  if (!canShow) {
    heatToken++
    heatProbs.value = null
    heatValue.value = null
    return
  }
  const token = ++heatToken
  let res: HeatmapResult | null = null
  try {
    res = await api.value!.get_heatmap()
  } catch {
    return
  }
  if (token !== heatToken) return // 过期结果丢弃
  if (res.error) return
  heatProbs.value = res.probs
  heatValue.value = res.value
}

// ---------- 对局流程 ----------
async function startGame() {
  if (!api.value) return
  busy.value = true
  notice.value = '开局中…'
  noticeSub.value = ''
  try {
    const res = await api.value.start_game(humanBlack.value)
    busy.value = false
    if (res.error) {
      notice.value = res.error
      noticeSub.value = '请先在上方选择模型'
      return
    }
    state.value = res
    humanBlack.value = res.humanBlack
    selectedSim.value = res.simulations
    renderStatus()
    await refreshHeatmap()
  } catch (e) {
    busy.value = false
    notice.value = '开局失败'
    noticeSub.value = String(e)
  }
}

async function switchModel() {
  if (!api.value) return
  busy.value = true
  notice.value = '模型加载中…'
  noticeSub.value = selectedModel.value
  try {
    await api.value.load_model(selectedModel.value)
  } catch (e) {
    busy.value = false
    notice.value = '模型加载失败'
    noticeSub.value = String(e)
    return
  }
  await startGame()
}

async function onHumanMove(r: number, c: number) {
  const s = state.value
  if (!api.value || !s || busy.value) return
  // 编辑模式：任何时候都可落子（黑白交替）；对局模式：仅轮到玩家时
  if (!editMode.value && (s.over || !s.humanTurn)) return
  if (s.board[r][c] !== 0) return
  busy.value = true
  heatProbs.value = null
  heatValue.value = null
  if (!editMode.value) {
    notice.value = 'AI 思考中…'
    noticeSub.value = `MCTS ${s.simulations} 次模拟`
  }
  try {
    const res = await api.value.human_move(r, c)
    if (res.error) {
      busy.value = false
      notice.value = res.error
      noticeSub.value = ''
      return
    }
    state.value = res
  } catch (e) {
    notice.value = '落子失败'
    noticeSub.value = String(e)
  }
  busy.value = false
  renderStatus()
  await refreshHeatmap()
}

// ---------- 编辑模式 ----------
async function toggleEdit() {
  if (!api.value) return
  busy.value = true
  notice.value = editMode.value ? '退出编辑模式…' : '进入编辑模式…'
  noticeSub.value = ''
  try {
    const res = await api.value.set_edit_mode(!editMode.value)
    if (res.error) {
      notice.value = res.error
      noticeSub.value = ''
      busy.value = false
      return
    }
    state.value = res
    // 关闭编辑模式时后端会重新分配黑白方（轮到方归 AI），同步执子身份
    humanBlack.value = res.humanBlack
  } catch (e) {
    notice.value = '切换编辑模式失败'
    noticeSub.value = String(e)
  }
  busy.value = false
  renderStatus()
  await refreshHeatmap()
}

async function clearBoard() {
  if (!api.value || !editMode.value) return
  try {
    const res = await api.value.clear_board()
    if (res.error) return
    state.value = res
  } catch {
    return
  }
  renderStatus()
}

function chooseColor(black: boolean) {
  if (humanBlack.value === black) return
  humanBlack.value = black
  startGame()
}

async function onSimChange() {
  if (!api.value) return
  try {
    const res = await api.value.set_simulations(selectedSim.value)
    if (res.error) {
      notice.value = res.error
    } else {
      selectedSim.value = res.simulations
    }
  } catch (e) {
    notice.value = '强度设置失败'
    noticeSub.value = String(e)
  }
}

async function toggleHeat() {
  heatOn.value = !heatOn.value
  await refreshHeatmap()
}

// ---------- 启动 ----------
onMounted(async () => {
  const bridge = await waitForBridge()
  if (!bridge) {
    bridgeMissing.value = true
    renderStatus()
    return
  }
  api.value = bridge
  try {
    models.value = await bridge.list_models()
  } catch {
    models.value = []
  }
  if (models.value.length) {
    selectedModel.value = models.value[models.value.length - 1] // 默认最新
    await switchModel()
  } else {
    notice.value = '未找到模型文件'
    noticeSub.value = '请先运行 python train.py 训练模型'
  }
})
</script>

<template>
  <div class="app">
    <div class="board-card">
      <GomokuBoard
        v-if="state"
        :board="state.board"
        :last-move="state.lastMove"
        :win-line="state.winLine"
        :over="state.over"
        :human-turn="state.humanTurn"
        :busy="busy"
        :heat-on="heatOn"
        :heat-probs="heatProbs"
        :hover-black="hoverBlack"
        :editable="editMode"
        @move="onHumanMove"
      />
      <div v-else class="board-placeholder">棋盘加载中…</div>
    </div>

    <div class="side">
      <div>
        <h1>五 子 棋</h1>
        <div class="sub">GOMOKU · CNN + MCTS 人机对弈</div>
      </div>

      <div class="card status">
        <div :class="['stone-dot', turnDotClass]"></div>
        <div class="status-text">
          <div class="status-main">{{ notice }}</div>
          <div class="status-sub">{{ noticeSub }}</div>
          <div v-if="winPct !== null" class="eval show">
            <div class="eval-label">
              <span>你的胜率</span><span>{{ winPct }}%</span><span>AI 胜率</span>
            </div>
            <div class="eval-track"><div class="eval-fill" :style="{ width: winPct + '%' }"></div></div>
          </div>
        </div>
      </div>

      <div class="card">
        <label class="field">模 型</label>
        <select v-model="selectedModel" @change="switchModel">
          <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
        </select>
        <label class="field field-gap">先 后 手（切换自动开局）</label>
        <div class="segment">
          <button :class="{ active: humanBlack }" @click="chooseColor(true)">你执黑 · 先手</button>
          <button :class="{ active: !humanBlack }" @click="chooseColor(false)">你执白 · 后手</button>
        </div>
        <label class="field field-gap">AI 强 度（模拟次数 · 越高越强越慢）</label>
        <select v-model.number="selectedSim" @change="onSimChange">
          <option v-for="n in SIM_CHOICES" :key="n" :value="n">{{ n }} 次 / 手</option>
        </select>
      </div>

      <button class="btn-primary" @click="startGame">新 游 戏</button>
      <div style="display: flex; gap: 8px;">
        <button :class="['btn-toggle', { active: editMode }]" @click="toggleEdit" style="flex: 1;">
          编 辑 模 式 ：{{ editMode ? '开' : '关' }}
        </button>
        <button v-if="editMode" class="btn-toggle" @click="clearBoard" style="flex: 1;">
          清 空 棋 盘
        </button>
      </div>
      <button :class="['btn-toggle', { active: heatOn }]" @click="toggleHeat">
        概 率 热 力 图 ：{{ heatOn ? '开' : '关' }}
      </button>
      <div class="hint">
        点击棋盘交叉点落子，任一方先连成五子即获胜。<br />
        编辑模式下自由交替黑白摆子，不触发 AI；关闭后 AI 从当前局面接管。<br />
        开启热力图后，绿圈越大表示 MCTS 越推荐该落子点。
      </div>
    </div>
  </div>
</template>
