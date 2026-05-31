import { useState, useRef, useCallback, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API = 'http://localhost:8000'

const EXAMPLES = [
  'Which visual categories drive the strongest FFA (face area) response?',
  'What visual content most engages the insula (threat/salience)?',
  'What visual content produces the most intense overall cortical engagement?',
]

interface StimResult {
  stimulus_id: string
  image_url?: string
  surface_url?: string
  target_value: number
  global_value: number
  peak_roi?: string
}

interface Category {
  index: number
  name: string
  rationale?: string
  images: { image_url?: string; stimulus_id: string }[]
  results: StimResult[]
  target_mean?: number
  global_mean?: number
  rank?: number
}

interface LogLine { id: number; text: string }

export function ExploreView() {
  const [question, setQuestion] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [targetRoi, setTargetRoi] = useState<string>('')
  const [targetRationale, setTargetRationale] = useState<string>('')
  const [categories, setCategories] = useState<Category[]>([])
  const [ranking, setRanking] = useState<any[] | null>(null)
  const [report, setReport] = useState<string | null>(null)
  const [phaseMsg, setPhaseMsg] = useState('')
  const [tribeDown, setTribeDown] = useState<string | null>(null)
  const [log, setLog] = useState<LogLine[]>([])
  const logId = useRef(0)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [log])
  const addLog = useCallback((t: string) => setLog(p => [...p, { id: logId.current++, text: t }]), [])

  const upCat = useCallback((idx: number, fn: (c: Category) => Category) => {
    setCategories(prev => { const i = prev.findIndex(c => c.index === idx); if (i < 0) return prev; const cp = [...prev]; cp[i] = fn(cp[i]); return cp })
  }, [])
  const upCatByName = useCallback((name: string, fn: (c: Category) => Category) => {
    setCategories(prev => { const i = prev.findIndex(c => c.name === name); if (i < 0) return prev; const cp = [...prev]; cp[i] = fn(cp[i]); return cp })
  }, [])

  const handle = useCallback((kind: string, d: any) => {
    switch (kind) {
      case 'phase': setPhaseMsg(d.message); addLog(d.message); break
      case 'ranking_designed':
        setTargetRoi(d.target_roi); setTargetRationale(d.target_roi_rationale)
        setCategories((d.categories || []).map((c: any, i: number) => ({
          index: i, name: c.name, rationale: c.rationale, images: [], results: [] })))
        addLog(`Target ROI: ${d.target_roi} · ${d.categories?.length} categories`); break
      case 'category_start': addLog(`Category: ${d.name}`); break
      case 'stimulus_generating': addLog(`  generating ${d.category}: "${d.prompt?.slice(0, 45)}"`); break
      case 'stimulus_ready':
        upCatByName(d.category, c => ({ ...c, images: [...c.images, { image_url: d.image_url, stimulus_id: d.stimulus_id }] })); break
      case 'stimulus_result':
        upCatByName(d.category, c => ({ ...c, results: [...c.results, d as StimResult] }))
        addLog(`  ${d.category} ${d.target_roi}=${d.target_value?.toFixed(3)} global=${d.global_value?.toFixed(3)}`); break
      case 'category_result':
        upCat(d.index, c => ({ ...c, target_mean: d.target_mean, global_mean: d.global_mean })); break
      case 'ranking':
        setRanking(d.ordered)
        d.ordered?.forEach((c: any) => upCatByName(c.name, x => ({ ...x, rank: c.rank })))
        addLog(`Ranking: ${d.ordered?.map((c: any) => c.name).join(' > ')}`); break
      case 'report': setReport(d.content); break
      case 'tribe_unavailable': setTribeDown(d.message); break
      case 'error': addLog(`Error: ${d.message}`); break
    }
  }, [addLog, upCat, upCatByName])

  const start = useCallback(async (q: string) => {
    if (!q.trim() || isRunning) return
    setIsRunning(true); setTargetRoi(''); setTargetRationale(''); setCategories([]); setRanking(null)
    setReport(null); setTribeDown(null); setLog([]); setPhaseMsg('Starting...')
    try {
      const res = await fetch(`${API}/api/explore?question=${encodeURIComponent(q)}`)
      const reader = res.body?.getReader(); const dec = new TextDecoder(); let buf = ''
      if (!reader) throw new Error('no body')
      while (true) {
        const { done, value } = await reader.read(); if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n'); buf = lines.pop() || ''
        let evt = ''
        for (const line of lines) {
          if (line.startsWith('event:')) evt = line.slice(6).trim()
          else if (line.startsWith('data:') && evt) { try { handle(evt, JSON.parse(line.slice(5).trim())) } catch {} ; evt = '' }
        }
      }
    } catch (e: any) { addLog(`Error: ${e.message}`) } finally { setIsRunning(false) }
  }, [isRunning, handle, addLog])

  const maxTarget = Math.max(0.001, ...categories.map(c => c.target_mean || 0))
  const started = categories.length > 0 || isRunning || !!tribeDown

  return (
    <div className="research-view">
      <div className="research-input-section">
        <h2>Generative Stimulus Exploration</h2>
        <p className="research-desc">
          Cortex designs controlled visual categories, generates real images, runs TRIBE v2,
          and ranks which content most engages the brain — by a target region and whole-cortex.
        </p>
        <div className="input-row">
          <input type="text" value={question} onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && start(question)}
            placeholder="What kind of visual content engages the brain most?"
            className="research-input" disabled={isRunning} />
          <button onClick={() => start(question)} disabled={isRunning || !question.trim()} className="research-btn">
            {isRunning ? <span className="spinner" /> : 'Explore'}
          </button>
        </div>
        {!started && (
          <div className="example-questions">
            {EXAMPLES.map(q => <button key={q} className="example-btn" onClick={() => { setQuestion(q); start(q) }} disabled={isRunning}>{q}</button>)}
          </div>
        )}
      </div>

      {tribeDown && <div className="banner banner-warn"><strong>TRIBE v2 not connected.</strong> {tribeDown}</div>}

      {started && (
        <div className="research-content">
          <div className="research-left">
            {targetRoi && (
              <div className="exp-card">
                <div className="exp-header"><span className="exp-roi">{targetRoi}</span>
                  <span className="exp-hyp">Target region for ranking</span></div>
                {targetRationale && <p className="exp-pred">{targetRationale}</p>}
              </div>
            )}

            {/* Live ranking */}
            {categories.some(c => c.target_mean != null) && (
              <div className="exp-card">
                <div className="iter-head">Engagement ranking · {targetRoi}</div>
                {[...categories].filter(c => c.target_mean != null)
                  .sort((a, b) => (b.target_mean || 0) - (a.target_mean || 0))
                  .map(c => (
                    <div key={c.index} className="rank-row">
                      <span className="rank-name">{c.rank ? `#${c.rank} ` : ''}{c.name}</span>
                      <div className="rank-bar-wrap">
                        <div className="rank-bar" style={{ width: `${(c.target_mean || 0) / maxTarget * 100}%` }} />
                      </div>
                      <span className="rank-val">{c.target_mean?.toFixed(3)}</span>
                      <span className="rank-global">global {c.global_mean?.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            )}

            {/* Category cards */}
            {categories.map(c => (
              <div key={c.index} className="exp-card">
                <div className="exp-header">
                  <span className="exp-hyp">{c.name}</span>
                  {c.target_mean != null && <span className="exp-iter">{targetRoi} {c.target_mean.toFixed(3)}</span>}
                </div>
                {c.rationale && <p className="exp-pred">{c.rationale}</p>}
                <div className="clip-grid">
                  {(c.results.length ? c.results : c.images).map((s: any, i: number) => (
                    <div key={i} className="clip-cell" title={s.peak_roi ? `peak ${s.peak_roi}` : ''}>
                      {s.image_url ? <img src={`${API}${s.image_url}`} className="clip-vid" alt="" />
                        : <div className="stim-placeholder"><span className="spinner-dark" /></div>}
                      {s.target_value != null && <span className="clip-val">{s.target_value.toFixed(2)}</span>}
                    </div>
                  ))}
                </div>
                {c.results.some((r: StimResult) => r.surface_url) && (
                  <div className="maps-stack">
                    {c.results.filter(r => r.surface_url).slice(0, 1).map((r, i) => (
                      <figure key={i} className="map-fig">
                        <figcaption>TRIBE v2 activation — {c.name}</figcaption>
                        <img className="surface-map" src={`${API}${r.surface_url}`} alt="" />
                      </figure>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {report && (
              <div className="report-card">
                <div className="report-header"><h3>Ranked Findings</h3><span className="report-badge">Complete</span></div>
                <div className="report-content markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown></div>
              </div>
            )}
          </div>

          <div className="research-right">
            <div className="brain-card">
              <div className="phase-head">
                <span className={`status-dot ${isRunning ? 'running' : 'done'}`} />
                <span className="phase-name">explore</span>
              </div>
              <p className="phase-msg">{phaseMsg}</p>
            </div>
            <div className="log-card">
              <h3>Live Log</h3>
              <div className="log" ref={logRef}>
                {log.map(l => <div key={l.id} className="log-line">{l.text}</div>)}
                {isRunning && <div className="log-line log-cursor">_</div>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
