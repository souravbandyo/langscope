import { Routes, Route } from 'react-router-dom'
import { PageLayout } from '@/components/layout/PageLayout'
import { ProtectedRoute } from '@/components/auth'
import { Home } from '@/pages/Home'
import { Dashboard } from '@/pages/Dashboard'
import { Rankings } from '@/pages/Rankings'
import { Arena } from '@/pages/Arena'
import { About } from '@/pages/About'
import { Recommendations } from '@/pages/Recommendations'
import { Specialists } from '@/pages/Specialists'
import { Benchmarks } from '@/pages/Benchmarks'
import { GroundTruth } from '@/pages/GroundTruth'
import { Transfer } from '@/pages/Transfer'
import { User } from '@/pages/User'
import { Models } from '@/pages/Models'
import { ModelDetail } from '@/pages/Models/ModelDetail'
import { BaseModels } from '@/pages/BaseModels'
import { Deployments } from '@/pages/Deployments'
import { SelfHosted } from '@/pages/SelfHosted'
import { MyModels } from '@/pages/MyModels'
import { PromptClassifier } from '@/pages/PromptClassifier'
import { AdminSettings } from '@/pages/AdminSettings'
import AuthPage from '@/pages/Auth'

function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/auth" element={<AuthPage />} />
      <Route path="/about" element={<PageLayout><About /></PageLayout>} />
      
      {/* Protected routes */}
      <Route element={<PageLayout />}>
        <Route path="/" element={
          <ProtectedRoute><Home /></ProtectedRoute>
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute><Dashboard /></ProtectedRoute>
        } />
        <Route path="/rankings" element={
          <ProtectedRoute><Rankings /></ProtectedRoute>
        } />
        <Route path="/rankings/:domain" element={
          <ProtectedRoute><Rankings /></ProtectedRoute>
        } />
        <Route path="/arena" element={
          <ProtectedRoute><Arena /></ProtectedRoute>
        } />
        <Route path="/recommendations" element={
          <ProtectedRoute><Recommendations /></ProtectedRoute>
        } />
        <Route path="/specialists" element={
          <ProtectedRoute><Specialists /></ProtectedRoute>
        } />
        <Route path="/benchmarks" element={
          <ProtectedRoute><Benchmarks /></ProtectedRoute>
        } />
        <Route path="/ground-truth" element={
          <ProtectedRoute><GroundTruth /></ProtectedRoute>
        } />
        <Route path="/transfer" element={
          <ProtectedRoute><Transfer /></ProtectedRoute>
        } />
        <Route path="/models" element={
          <ProtectedRoute><Models /></ProtectedRoute>
        } />
        <Route path="/models/*" element={
          <ProtectedRoute><ModelDetail /></ProtectedRoute>
        } />
        <Route path="/base-models" element={
          <ProtectedRoute><BaseModels /></ProtectedRoute>
        } />
        <Route path="/deployments" element={
          <ProtectedRoute><Deployments /></ProtectedRoute>
        } />
        <Route path="/self-hosted" element={
          <ProtectedRoute><SelfHosted /></ProtectedRoute>
        } />
        <Route path="/my-models" element={
          <ProtectedRoute><MyModels /></ProtectedRoute>
        } />
        <Route path="/prompt-classifier" element={
          <ProtectedRoute><PromptClassifier /></ProtectedRoute>
        } />
        <Route path="/admin" element={
          <ProtectedRoute><AdminSettings /></ProtectedRoute>
        } />
        <Route path="/user" element={
          <ProtectedRoute><User /></ProtectedRoute>
        } />
      </Route>
    </Routes>
  )
}

export default App
