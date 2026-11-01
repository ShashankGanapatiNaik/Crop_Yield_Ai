import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg'],
      manifest: {
        name: 'Crop Yield AI',
        short_name: 'CropYield',
        description: 'AI-based crop yield prediction and decision support for farmers',
        theme_color: '#14472f',
        background_color: '#f3f6ef',
        display: 'standalone',
        start_url: '/',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        navigateFallbackDenylist: [/^\/api\//],
        runtimeCaching: [
          {
            urlPattern: ({ url }) => /\/api\/(crops|model-info)$/.test(url.pathname),
            handler: 'NetworkFirst',
            options: { cacheName: 'api-reference', networkTimeoutSeconds: 4 },
          },
          {
            urlPattern: ({ url }) => url.origin.includes('fonts.g'),
            handler: 'StaleWhileRevalidate',
            options: { cacheName: 'fonts' },
          },
        ],
      },
    }),
  ],
  server: {
    port: 5173,
    proxy: { '/api': { target: process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') } },
  },
})
