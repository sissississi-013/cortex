import { useState, useCallback } from 'react'
import { BrainVisualization } from './BrainVisualization'

const API = 'http://localhost:8000'

type ToolId = 'predict_fmri' | 'search_literature' | 'meta_analyze' | 'decode_brain_region' | 'analyze_eeg'

interface ToolConfig {
  id: ToolId
  name: string
  description: string
  icon: string
  defaultInput: Record<string, string>
}

const TOOLS: ToolConfig[] = [
  {
    id: 'predict_fmri',
    name: 'predict_fmri',
    description: 'Predict brain activation for any stimulus using TRIBE v2',
    icon: '01',
    defaultInput: { stimulus: 'a person listening to classical piano music' },
  },
  {
    id: 'search_literature',
    name: 'search_literature',
    description: 'Search 15,000+ neuroimaging studies in Neurosynth',
    icon: '02',
    defaultInput: { query: 'speech perception' },
  },
  {
    id: 'meta_analyze',
    name: 'meta_analyze',
    description: 'Run ALE coordinate-based meta-analysis',
    icon: '03',
    defaultInput: { term: 'fear' },
  },
  {
    id: 'decode_brain_region',
    name: 'decode_brain_region',
    description: 'Decode cognitive functions from MNI coordinates',
    icon: '04',
    defaultInput: { x: '-22', y: '-4', z: '-18' },
  },
  {
    id: 'analyze_eeg',
    name: 'analyze_eeg',
    description: 'Analyze EEG data — ERP, time-frequency, source localization',
    icon: '05',
    defaultInput: { analysis_type: 'erp', condition: 'auditory' },
  },
]

export function ToolExplorer() {
  const [activeTool, setActiveTool] = useState<ToolId>('predict_fmri')
  const [inputs, setInputs] = useState<Record<string, string>>(TOOLS[0].defaultInput)
  // Tool responses are intentionally heterogeneous across MCP-like endpoints.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [brainRegions, setBrainRegions] = useState<any[]>([])

  const runTool = useCallback(async () => {
    setLoading(true)
    setResult(null)
    setBrainRegions([])

    try {
      const params = new URLSearchParams(inputs)
      const res = await fetch(`${API}/api/tools/${activeTool}?${params}`)
      const data = await res.json()
      setResult(data)

      if (data.top_activated_regions) {
        setBrainRegions(data.top_activated_regions)
      }
    } catch (e: unknown) {
      setResult({ error: e instanceof Error ? e.message : String(e) })
    } finally {
      setLoading(false)
    }
  }, [activeTool, inputs])

  const selectTool = (tool: ToolConfig) => {
    setActiveTool(tool.id)
    setInputs(tool.defaultInput)
    setResult(null)
    setBrainRegions([])
  }

  return (
    <div className="tool-explorer">
      <div className="tool-sidebar">
        <div className="tool-sidebar-label">Agent interface</div>
        <h3>Brain Toolchain</h3>
        <p className="tool-sidebar-desc">
          Production-facing capabilities exposed to any MCP-compatible research agent.
        </p>
        {TOOLS.map((tool) => (
          <button
            key={tool.id}
            className={`tool-card ${activeTool === tool.id ? 'active' : ''}`}
            onClick={() => selectTool(tool)}
          >
            <span className="tool-icon">{tool.icon}</span>
            <div>
              <span className="tool-name">{tool.name}</span>
              <span className="tool-desc">{tool.description}</span>
            </div>
          </button>
        ))}
      </div>

      <div className="tool-main">
        <div className="tool-input-section">
          <h3>{activeTool}</h3>
          <div className="tool-inputs">
            {Object.entries(inputs).map(([key, value]) => (
              <div key={key} className="tool-input-row">
                <label>{key}</label>
                <input
                  type="text"
                  value={value}
                  onChange={(e) => setInputs(prev => ({ ...prev, [key]: e.target.value }))}
                  onKeyDown={(e) => e.key === 'Enter' && runTool()}
                />
              </div>
            ))}
          </div>
          <button onClick={runTool} disabled={loading} className="run-btn">
            {loading ? <span className="spinner" /> : 'Run Tool'}
          </button>
        </div>

        {brainRegions.length > 0 && (
          <BrainVisualization
            regions={brainRegions}
            title="Predicted Brain Activation"
          />
        )}

        {result && (
          <div className="tool-result">
            <h4>Result</h4>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  )
}
