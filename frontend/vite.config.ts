import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// 固定 5173 端口：play.py 中的 pywebview 窗口硬编码加载该地址。
// base: './' 使构建产物使用相对路径引用资源，
// 便于 pywebview 内置静态服务器（及任意子路径部署）加载。
export default defineConfig({
  plugins: [vue()],
  base: './',
  server: {
    port: 5173,
    strictPort: true,
  },
})
