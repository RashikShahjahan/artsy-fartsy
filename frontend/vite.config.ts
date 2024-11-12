import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // Ensure this is set
  build: {
    outDir: 'dist', // This should be 'dist', not '../dist'
    emptyOutDir: true
  }
})
