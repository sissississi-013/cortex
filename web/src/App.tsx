import { useState } from 'react'
import './App.css'
import { ResearchView } from './components/ResearchView'
import { ToolExplorer } from './components/ToolExplorer'

type Tab = 'research' | 'tools'

function App() {
  const [tab, setTab] = useState<Tab>('research')

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <img src="/cortex-mark.png" alt="" className="logo-mark" />
            <h1>Cortex</h1>
          </div>
          <span className="subtitle">Autoresearch for brain systems</span>
        </div>
        <nav className="tabs">
          <button className={tab === 'research' ? 'active' : ''} onClick={() => setTab('research')}>
            Research Agent
          </button>
          <button className={tab === 'tools' ? 'active' : ''} onClick={() => setTab('tools')}>
            Brain Tools
          </button>
        </nav>
        <div className="header-status">
          <span className="status-light" />
          Live demo build
        </div>
      </header>
      <main className="main">
        {tab === 'research' ? <ResearchView /> : <ToolExplorer />}
      </main>
    </div>
  )
}

export default App
