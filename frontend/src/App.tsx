import { lazy, Suspense, type ReactNode } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuth } from './auth'
import Layout from './components/Layout'
import { ErrorBox, Skeleton } from './components/ui'
import Home from './pages/Home'
import Login from './pages/Login'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const Predict = lazy(() => import('./pages/Predict'))
const Result = lazy(() => import('./pages/Result'))
const WhatIf = lazy(() => import('./pages/WhatIf'))
const Compare = lazy(() => import('./pages/Compare'))
const History = lazy(() => import('./pages/History'))
const Insights = lazy(() => import('./pages/Insights'))
const CropHealth = lazy(() => import('./pages/CropHealth'))
const Admin = lazy(() => import('./pages/Admin'))

function Guard({ children, admin }: { children: ReactNode; admin?: boolean }) {
  const { user, loading } = useAuth()
  const { t } = useTranslation()
  if (loading) return <Skeleton className="h-40 w-full" />
  if (!user) return <Navigate to="/login" replace />
  if (admin && user.role !== 'admin') return <ErrorBox message={t('admin.forbidden')} />
  return <>{children}</>
}

export default function App() {
  const { t } = useTranslation()
  return (
    <Suspense fallback={<p className="p-8 text-muted" role="status">{t('common.loading')}</p>}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="login" element={<Login mode="login" />} />
          <Route path="register" element={<Login mode="register" />} />
          <Route path="insights" element={<Insights />} />
          <Route path="dashboard" element={<Guard><Dashboard /></Guard>} />
          <Route path="predict" element={<Guard><Predict /></Guard>} />
          <Route path="result/:id" element={<Guard><Result /></Guard>} />
          <Route path="whatif" element={<Guard><WhatIf /></Guard>} />
          <Route path="compare" element={<Guard><Compare /></Guard>} />
          <Route path="history" element={<Guard><History /></Guard>} />
          <Route path="crop-health" element={<Guard><CropHealth /></Guard>} />
          <Route path="admin" element={<Guard admin><Admin /></Guard>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}