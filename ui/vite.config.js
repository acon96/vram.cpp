import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],

  // Use './' as base so the built app works from any subdirectory
  // (e.g. GitHub Pages project site at /vram.cpp/).
  // Override with VITE_BASE_URL env var for a specific deployment path.
  base: process.env.VITE_BASE_URL ?? './',

  build: {
    outDir: 'dist',
    // Allow the WASM files (loaded as external scripts) to be slightly large
    chunkSizeWarningLimit: 4096,
  },

  server: {
    // During dev, proxy wasm/ to the vendor build output directory.
    // Run: VITE_WASM_BASE_URL=../build-wasm-vendor npm run dev
    // or configure fs.allow to reach outside the ui/ directory.
    fs: {
      allow: ['..'],
    },
  },
})
