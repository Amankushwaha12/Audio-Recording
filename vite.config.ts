import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import {defineConfig} from 'vite';

function deepFilterWasmPlugin() {
  const srcWasm = path.resolve(
    __dirname,
    'node_modules/@lofcz/deepfilternet-web/dist/df_bg.wasm'
  );
  let cachedGz: Buffer | null = null;

  const getCompressedWasm = () => {
    if (!cachedGz && fs.existsSync(srcWasm)) {
      cachedGz = zlib.gzipSync(fs.readFileSync(srcWasm), {level: 6});
    }
    return cachedGz;
  };

  return {
    name: 'deepfilter-wasm-compressed',
    configureServer(server: {middlewares: {use: Function}}) {
      server.middlewares.use('/df_bg.wasm.bin', (_req: unknown, res: any, next: Function) => {
        const gz = getCompressedWasm();
        if (!gz) {
          next();
          return;
        }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Length', String(gz.byteLength));
        res.setHeader('Cache-Control', 'public, max-age=3600');
        res.end(gz);
      });
    },
    closeBundle(error?: Error): void {
      if (error) return;
      const distDir = path.resolve(__dirname, 'dist');
      const gz = getCompressedWasm();
      if (gz && fs.existsSync(distDir)) {
        fs.writeFileSync(path.join(distDir, 'df_bg.wasm.bin'), gz);
      }
    },
  };
}

export default defineConfig(() => {
  return {
    plugins: [
      react(),
      tailwindcss(),
      deepFilterWasmPlugin(),
      // Copies src/assets into dist/ so root-relative URLs like
      // "/src/assets/images/..." (e.g. generate_image output) still resolve
      // in production builds. Do not remove.
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(__dirname, 'dist');
          const srcAssets = path.resolve(__dirname, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, {recursive: true, force: true});
          }
        },
      },
    ],
    optimizeDeps: {
      exclude: ['@lofcz/deepfilternet-web'],
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
