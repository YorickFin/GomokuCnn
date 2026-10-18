/**
 * pywebview JS 桥封装。
 *
 * 桌面窗口由 play.py 创建，pywebview 会向页面注入 window.pywebview.api；
 * 在普通浏览器中直接打开时桥不存在，waitForBridge 会返回 null。
 */
import type { HeatmapResult, StateResult } from './types'

interface PywebviewApi {
  list_models(): Promise<string[]>
  load_model(name: string): Promise<{ ok: boolean; name: string }>
  start_game(human_black: boolean): Promise<StateResult>
  human_move(r: number, c: number): Promise<StateResult>
  get_heatmap(): Promise<HeatmapResult>
  set_simulations(n: number): Promise<{ ok: boolean; simulations: number; error?: string }>
  set_edit_mode(on: boolean): Promise<StateResult>
  clear_board(): Promise<StateResult>
}

declare global {
  interface Window {
    pywebview?: { api: PywebviewApi }
  }
}

/** 等待 pywebview 桥就绪；超时（普通浏览器环境）返回 null。 */
export function waitForBridge(timeoutMs = 12000): Promise<PywebviewApi | null> {
  return new Promise((resolve) => {
    if (window.pywebview?.api) {
      resolve(window.pywebview.api)
      return
    }
    const timer = window.setTimeout(() => resolve(null), timeoutMs)
    window.addEventListener(
      'pywebviewready',
      () => {
        window.clearTimeout(timer)
        resolve(window.pywebview?.api ?? null)
      },
      { once: true },
    )
  })
}
