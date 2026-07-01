import { defineConfig } from 'vite';
import { resolve } from 'path';

// The Gaussian-splat renderer runs its sort in a web worker; with
// sharedMemoryForWorkers:false (set in main.js/raw.js) no COOP/COEP headers are needed.
export default defineConfig({
  server: { host: true, port: 10100, allowedHosts: true },
  optimizeDeps: { include: ['three', '@mkkellogg/gaussian-splats-3d'] },
  // scene.ply/scene.ksplat live in public/ and are served as-is (not bundled).
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        raw: resolve(__dirname, 'raw.html'),   // minimal standalone splat-file viewer
      },
    },
  },
});
