import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // Add this line
  build: {
    outDir: '../dist',  // Output to parent dist folder
  }
})
