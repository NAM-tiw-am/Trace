import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// Use VITE_OUT_DIR env to override output dir.
// Locally (serving via FastAPI): set VITE_OUT_DIR=../static
// On Vercel: leave unset → defaults to 'dist'
const outDir = process.env.VITE_OUT_DIR ?? 'dist';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/',
  build: {
    outDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/storage': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
