import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],

  // Use './' as base so the built app works from any subdirectory
  // (for example, a GitHub Pages project site).
  // Override with VITE_BASE_URL env var for a specific deployment path.
  base: process.env.VITE_BASE_URL ?? './',

  build: {
    outDir: 'dist',
    // Allow the WASM files (loaded as external scripts) to be slightly large
    chunkSizeWarningLimit: 4096,
  },
})
