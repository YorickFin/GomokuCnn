<script setup lang="ts">
/**
 * 五子棋棋盘：Canvas 绘制木纹棋盘、棋子、最后一手标记、
 * 获胜连线高亮与 MCTS 概率热力图叠加层。
 * 所有叠加层统一在 draw() 中渲染，避免被重绘覆盖。
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const N = 15
const STAR_POINTS: [number, number][] = [
  [3, 3], [3, 11], [7, 7], [11, 3], [11, 11],
]

const props = defineProps<{
  board: number[][]
  lastMove: [number, number] | null
  winLine: [number, number][]
  over: boolean
  humanTurn: boolean
  busy: boolean
  heatOn: boolean
  heatProbs: number[][] | null
  /** 悬停预览颜色：对局时=玩家棋子色，编辑模式=当前手数应落色 */
  hoverBlack: boolean
  /** 编辑模式：允许悬停预览（不受 humanTurn 限制） */
  editable: boolean
}>()

const emit = defineEmits<{
  (e: 'move', r: number, c: number): void
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let ctx: CanvasRenderingContext2D | null = null
let resizeObserver: ResizeObserver | null = null
let hover: [number, number] | null = null

// ---------- 几何 ----------
function geom() {
  const size = canvasRef.value?.clientWidth ?? 560
  const margin = size * 0.058
  const cell = (size - 2 * margin) / (N - 1)
  return { size, margin, cell }
}
function toXY(r: number, c: number): [number, number] {
  const { margin, cell } = geom()
  return [margin + c * cell, margin + r * cell]
}
function toCell(x: number, y: number): [number, number] | null {
  const { margin, cell } = geom()
  const c = Math.round((x - margin) / cell)
  const r = Math.round((y - margin) / cell)
  if (r < 0 || r >= N || c < 0 || c >= N) return null
  const [px, py] = toXY(r, c)
  if (Math.hypot(x - px, y - py) > cell * 0.45) return null
  return [r, c]
}

// ---------- 绘制 ----------
function drawStone(x: number, y: number, r: number, kind: number) {
  if (!ctx) return
  const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r)
  if (kind === 1) {
    g.addColorStop(0, '#6a6a6a')
    g.addColorStop(1, '#0b0b0b')
  } else {
    g.addColorStop(0, '#ffffff')
    g.addColorStop(1, '#c7c7c7')
  }
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fillStyle = g
  ctx.fill()
  if (kind === -1) {
    ctx.strokeStyle = 'rgba(120,110,90,0.55)'
    ctx.lineWidth = 1
    ctx.stroke()
  }
}

function draw() {
  if (!ctx) return
  const { size, margin, cell } = geom()
  ctx.clearRect(0, 0, size, size)

  // 棋盘木色
  const wood = ctx.createLinearGradient(0, 0, size, size)
  wood.addColorStop(0, '#ecc88a')
  wood.addColorStop(1, '#dcae63')
  ctx.fillStyle = wood
  ctx.fillRect(0, 0, size, size)

  // 网格线
  ctx.strokeStyle = '#8a5a2b'
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let i = 0; i < N; i++) {
    const p = margin + i * cell
    ctx.moveTo(margin, p)
    ctx.lineTo(margin + (N - 1) * cell, p)
    ctx.moveTo(p, margin)
    ctx.lineTo(p, margin + (N - 1) * cell)
  }
  ctx.stroke()
  ctx.lineWidth = 2.2
  ctx.strokeRect(margin, margin, (N - 1) * cell, (N - 1) * cell)

  // 星位
  ctx.fillStyle = '#7a4a1e'
  for (const [r, c] of STAR_POINTS) {
    const [x, y] = toXY(r, c)
    ctx.beginPath()
    ctx.arc(x, y, Math.max(3, cell * 0.11), 0, Math.PI * 2)
    ctx.fill()
  }

  const rStone = cell * 0.46

  // 概率热力图（仅空位、对局进行中）：绿圈大小/深浅 ∝ MCTS 访问概率
  if (props.heatOn && props.heatProbs && !props.over) {
    let pmax = 1e-9
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        if (props.board[r][c] === 0 && props.heatProbs[r][c] > pmax) {
          pmax = props.heatProbs[r][c]
        }
      }
    }
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const p = props.heatProbs[r][c]
        if (props.board[r][c] !== 0 || p <= 0) continue
        const t = p / pmax // 按本局最大值归一化
        if (t < 0.06) continue
        const [x, y] = toXY(r, c)
        const rr = rStone * (0.28 + 0.72 * t)
        ctx.beginPath()
        ctx.arc(x, y, rr, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(46, 125, 91, ${(0.1 + 0.6 * t).toFixed(3)})`
        ctx.fill()
        if (t >= 0.55) {
          ctx.fillStyle = '#1d4d33'
          ctx.font = `600 ${Math.round(cell * 0.3)}px "Microsoft YaHei UI", sans-serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(`${Math.round(p * 100)}%`, x, y)
        }
      }
    }
  }

  // 悬停预览
  const canHover =
    hover && !props.over && (props.humanTurn || props.editable) && !props.busy &&
    props.board[hover[0]][hover[1]] === 0
  if (hover && canHover) {
    const [x, y] = toXY(hover[0], hover[1])
    ctx.beginPath()
    ctx.arc(x, y, rStone, 0, Math.PI * 2)
    ctx.fillStyle = props.hoverBlack ? 'rgba(20,20,20,0.25)' : 'rgba(255,255,255,0.5)'
    ctx.fill()
    ctx.strokeStyle = '#3e7c5b'
    ctx.lineWidth = 1.5
    ctx.stroke()
  }

  // 棋子
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const v = props.board[r][c]
      if (v !== 0) {
        const [x, y] = toXY(r, c)
        drawStone(x, y, rStone, v)
      }
    }
  }

  // 最后一手标记
  if (props.lastMove && !props.over) {
    const [x, y] = toXY(props.lastMove[0], props.lastMove[1])
    ctx.beginPath()
    ctx.arc(x, y, Math.max(3, cell * 0.13), 0, Math.PI * 2)
    ctx.fillStyle = '#d64541'
    ctx.fill()
  }

  // 获胜五子高亮
  if (props.over && props.winLine.length) {
    ctx.strokeStyle = '#d64541'
    ctx.lineWidth = 3
    for (const [r, c] of props.winLine) {
      const [x, y] = toXY(r, c)
      ctx.beginPath()
      ctx.arc(x, y, rStone + 2.5, 0, Math.PI * 2)
      ctx.stroke()
    }
  }
}

function resize() {
  const canvas = canvasRef.value
  if (!canvas || !ctx) return
  const dpr = window.devicePixelRatio || 1
  const size = canvas.clientWidth
  canvas.width = size * dpr
  canvas.height = size * dpr
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  draw()
}

// ---------- 交互 ----------
function onClick(e: MouseEvent) {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const cell = toCell(e.clientX - rect.left, e.clientY - rect.top)
  if (cell) emit('move', cell[0], cell[1])
}

function onMouseMove(e: MouseEvent) {
  const canvas = canvasRef.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const cell = toCell(e.clientX - rect.left, e.clientY - rect.top)
  const key = cell ? `${cell[0]},${cell[1]}` : null
  const old = hover ? `${hover[0]},${hover[1]}` : null
  if (key !== old) {
    hover = cell
    draw()
  }
}

function onMouseLeave() {
  hover = null
  draw()
}

onMounted(() => {
  ctx = canvasRef.value?.getContext('2d') ?? null
  if (canvasRef.value) {
    resizeObserver = new ResizeObserver(() => resize())
    resizeObserver.observe(canvasRef.value)
  }
  resize()
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
})

// 任意 props 变化后重绘（棋盘/热力图/状态均不可变替换，deep 保证数组更新触发）
watch(
  () => [
    props.board, props.heatProbs, props.lastMove, props.winLine,
    props.over, props.humanTurn, props.busy, props.heatOn,
    props.hoverBlack, props.editable,
  ],
  () => draw(),
  { deep: true, flush: 'post' },
)
</script>

<template>
  <canvas
    ref="canvasRef"
    class="board-canvas"
    @click="onClick"
    @mousemove="onMouseMove"
    @mouseleave="onMouseLeave"
  ></canvas>
</template>

<style scoped>
.board-canvas {
  display: block;
  width: min(58vh, 560px);
  height: min(58vh, 560px);
  border-radius: 10px;
  cursor: pointer;
}
</style>
