import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { registerSW } from 'virtual:pwa-register'
import './i18n'
import './index.css'
import App from './App'
import { AuthProvider } from './auth'
import { ThemeProvider } from './theme'
import { ToastProvider } from './toast'

registerSW({ immediate: true })

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ThemeProvider><ToastProvider><AuthProvider><App /></AuthProvider></ToastProvider></ThemeProvider>
    </BrowserRouter>
  </StrictMode>,
)
