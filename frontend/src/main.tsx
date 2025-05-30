import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App'
import DrawingPage from './pages/DrawingPage'
import SearchPage from './pages/SearchPage'
import { AnalyticsProvider } from 'rashik-analytics-provider';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <AnalyticsProvider 
        serviceName="artsy"
        endpoint="https://analytics.rashik.sh/api"
      >
        <Routes>
          <Route path="/" element={<App />}>
            <Route index element={<DrawingPage />} />
            <Route path="search" element={<SearchPage />} />
          </Route>
        </Routes>
      </AnalyticsProvider>
    </BrowserRouter>
  </StrictMode>
)
