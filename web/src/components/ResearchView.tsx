import { useState, useRef, useCallback, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API = 'http://localhost:8000'

const EXAMPLE_QUESTIONS = [
  'Does the FFA respond more to faces than to houses?',
  'Does the STG respond more to speech than to music?',
  'Does V1 respond more to high- vs low-complexity scenes?',
  'Does motor cortex activate more for action vs static videos?',
]

interface Datapoint {
  condition: string
  stimulus_id: string
  roi: string
  value: number
  iteration: number
  peak_roi?: string
  caption?: string
  url?: string
  surface_url?: string
  roi_means?: Record<string, number>
}

interface Stimulus {
  condition: string
  stimulus_id: string
  url?: string
  status: 'generating' | 'ready' | 'tribe' | 'done' | 'failed'
  prompt?: string
  caption?: string
  similarity?: number
}

interface IterRecord {
  iteration: number
  stats?: any
  decision?: any
  maps?: { group_a_url?: string; group_b_url?: string; contrast_url?: string; group_a_label?: string; group_b_label?: string; roi_meta?: any }
  refinement?: { diagnosis?: string; prompts_a?: string[]; prompts_b?: string[] }
}

interface Experiment {
  index: number
  hypothesis: string
  focal_roi: string
  prediction?: string
  condition_a?: string
  condition_b?: string
  stop_condition?: string
  iteration: number
  stimuli: Stimulus[]
  datapoints: Datapoint[]
  iterations: IterRecord[]
  complete: boolean
  status?: string
}

interface LogLine { id: number; text: string; kind: string }

export function ResearchView() {
  const [question, setQuestion] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [phase, setPhase] = useState('')
  const [phaseMsg, setPhaseMsg] = useState('')
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [report, setReport] = useState<string | null>(null)
  const [tribeUnavailable, setTribeUnavailable] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [log, setLog] = useState<LogLine[]>([])
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const logId = useRef(0)

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  const addLog = useCallback((text: string, kind = 'info') => {
    setLog(prev => [...prev, { id: logId.current++, text, kind }])
  }, [])

  const updateExp = useCallback((index: number, fn: (e: Experiment) => Experiment) => {
    setExperiments(prev => {
      const i = prev.findIndex(e => e.index === index)
      if (i < 0) return prev
      const copy = [...prev]; copy[i] = fn(copy[i]); return copy
    })
  }, [])

  const handleEvent = useCallback((kind: string, d: any) => {
    switch (kind) {
      case 'phase':
        setPhase(d.phase); setPhaseMsg(d.message); addLog(d.message, 'phase'); break
      case 'experiments_designed':
        addLog(`Designed ${d.count} experiment(s)`, 'phase'); break
      case 'experiment_start':
        setExperiments(prev => [...prev, {
          index: d.index, hypothesis: d.hypothesis, focal_roi: d.focal_roi,
          prediction: d.prediction, condition_a: d.condition_a, condition_b: d.condition_b,
          stop_condition: d.stop_condition,
          iteration: 0, stimuli: [], datapoints: [], iterations: [], complete: false,
        }])
        addLog(`Experiment ${d.index + 1}: ${d.hypothesis}`, 'exp'); break
      case 'iteration_start':
        updateExp(d.experiment, e => ({
          ...e, iteration: d.iteration,
          iterations: e.iterations.some(i => i.iteration === d.iteration)
            ? e.iterations : [...e.iterations, { iteration: d.iteration }],
        }))
        addLog(`  Iteration ${d.iteration} — ${d.n_per_condition} clips/condition (parallel TRIBE)`, 'iter'); break
      case 'datapoint':
        updateExp(d.experiment, e => ({ ...e, datapoints: [...e.datapoints, d as Datapoint] }))
        addLog(`    ${d.condition} ${d.roi}=${d.value?.toFixed(3)} (peak ${d.peak_roi})`, 'data'); break
      case 'maps':
        updateExp(d.experiment, e => ({
          ...e, iterations: e.iterations.map(i => i.iteration === d.iteration ? { ...i, maps: {
            group_a_url: d.group_a_url, group_b_url: d.group_b_url, contrast_url: d.contrast_url,
            group_a_label: d.group_a_label, group_b_label: d.group_b_label, roi_meta: d.roi_meta } } : i),
        }))
        addLog(`    Rendered group + contrast cortical maps`, 'tribe'); break
      case 'stats':
        updateExp(d.experiment, e => ({
          ...e, iterations: e.iterations.map(i => i.iteration === d.iteration ? { ...i, stats: d.result } : i),
        }))
        if (d.result?.p_value != null)
          addLog(`  Stats(it${d.iteration}): ${d.result.direction}, p=${d.result.p_value.toFixed(3)}, d=${d.result.cohens_d?.toFixed(2)} (n=${d.result.n_a}/${d.result.n_b})`, 'stats')
        break
      case 'decision':
        updateExp(d.experiment, e => ({
          ...e, status: d.status,
          iterations: e.iterations.map(i => i.iteration === d.iteration ? { ...i, decision: d } : i),
        }))
        addLog(`  ${d.status}: ${d.stop_reason}`, 'decision'); break
      case 'refinement':
        updateExp(d.experiment, e => ({
          ...e, iterations: e.iterations.map(i => i.iteration === d.iteration ? { ...i, refinement: d } : i),
        }))
        addLog(`  Refine: ${d.diagnosis?.slice(0, 70)}`, 'decision'); break
      case 'experiment_complete':
        updateExp(d.index, e => ({ ...e, complete: true })); addLog(`Experiment ${d.index + 1} complete`, 'phase'); break
      case 'tribe_unavailable':
        setTribeUnavailable(d.message); addLog(`TRIBE unavailable: ${d.message}`, 'error'); break
      case 'report':
        setReport(d.content); break
      case 'error':
        setErrorMsg(d.message); addLog(`Error: ${d.message}`, 'error'); break
    }
  }, [addLog, updateExp])

  const startResearch = useCallback(async (q: string) => {
    if (!q.trim() || isRunning) return
    setIsRunning(true); setPhase('design'); setPhaseMsg('Starting...')
    setExperiments([]); setReport(null); setTribeUnavailable(null); setErrorMsg(null); setLog([]); setElapsed(0)
    const start = Date.now()
    timerRef.current = setInterval(() => setElapsed(Date.now() - start), 200)
    try {
      const res = await fetch(`${API}/api/research?question=${encodeURIComponent(q)}`)
      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      if (!reader) throw new Error('No response body')
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        let evt = ''
        for (const line of lines) {
          if (line.startsWith('event:')) evt = line.slice(6).trim()
          else if (line.startsWith('data:') && evt) {
            try { handleEvent(evt, JSON.parse(line.slice(5).trim())) } catch {}
            evt = ''
          }
        }
      }
    } catch (e: any) {
      setErrorMsg(e.message)
    } finally {
      setIsRunning(false)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRunning, handleEvent])

  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000)
    return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`
  }

  const hasStarted = experiments.length > 0 || isRunning || !!errorMsg || !!tribeUnavailable

  return (
    <div className="research-view">
      <div className="research-input-section">
        <h2>Autonomous Neuroscience Research</h2>
        <p className="research-desc">
          Cortex designs experiments, retrieves real video stimuli, runs TRIBE v2 on GPU,
          measures cortical activation, and iterates to a statistical conclusion.
        </p>
        <div className="input-row">
          <input type="text" value={question} onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && startResearch(question)}
            placeholder="Ask a falsifiable question about a brain region..."
            className="research-input" disabled={isRunning} />
          <button onClick={() => startResearch(question)} disabled={isRunning || !question.trim()} className="research-btn">
            {isRunning ? <span className="spinner" /> : 'Research'}
          </button>
        </div>
        {!hasStarted && (
          <div className="example-questions">
            {EXAMPLE_QUESTIONS.map(q => (
              <button key={q} className="example-btn" onClick={() => { setQuestion(q); startResearch(q) }} disabled={isRunning}>{q}</button>
            ))}
          </div>
        )}
      </div>

      {tribeUnavailable && (
        <div className="banner banner-warn">
          <strong>TRIBE v2 not connected.</strong> {tribeUnavailable}
          <div className="banner-sub">No activation data is fabricated — connect Modal to run real inference.</div>
        </div>
      )}
      {errorMsg && !tribeUnavailable && <div className="banner banner-error"><strong>Error:</strong> {errorMsg}</div>}

      {hasStarted && (
        <div className="research-content">
          <div className="research-left">
            {experiments.map(exp => <ExperimentCard key={exp.index} exp={exp} />)}
            {report && (
              <div className="report-card">
                <div className="report-header"><h3>Research Report</h3><span className="report-badge">Complete</span></div>
                <div className="report-content markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
          <div className="research-right">
            <div className="brain-card">
              <div className="phase-head">
                <span className={`status-dot ${isRunning ? 'running' : 'done'}`} />
                <span className="phase-name">{phase || 'idle'}</span>
                <span className="phase-elapsed">{fmtTime(elapsed)}</span>
              </div>
              <p className="phase-msg">{phaseMsg}</p>
            </div>
            <div className="log-card">
              <h3>Live Log</h3>
              <div className="log" ref={logRef}>
                {log.map(l => <div key={l.id} className={`log-line log-${l.kind}`}>{l.text}</div>)}
                {isRunning && <div className="log-line log-cursor">_</div>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ExperimentCard({ exp }: { exp: Experiment }) {
  const condA = exp.condition_a || 'A'
  const condB = exp.condition_b || 'B'

  return (
    <div className="exp-card">
      <div className="exp-header">
        <span className="exp-roi">{exp.focal_roi}</span>
        <span className="exp-hyp">{exp.hypothesis}</span>
        {exp.complete ? <span className={`exp-status ${exp.status}`}>{exp.status}</span>
          : <span className="exp-iter">iter {exp.iteration}</span>}
      </div>
      {exp.prediction && <p className="exp-pred">{exp.prediction}</p>}
      {exp.stop_condition && <p className="exp-stop">Stop condition: {exp.stop_condition}</p>}

      {/* Per-iteration research trace */}
      {exp.iterations.map(it => {
        const itDpA = exp.datapoints.filter(d => d.iteration === it.iteration && d.condition === condA)
        const itDpB = exp.datapoints.filter(d => d.iteration === it.iteration && d.condition === condB)
        const roiMeta = it.maps?.roi_meta
        return (
          <div key={it.iteration} className="iter-block">
            <div className="iter-head">Iteration {it.iteration} · {itDpA.length + itDpB.length} clips · TRIBE v2 (parallel)</div>

            {/* Real group-average + contrast cortical surface maps */}
            {it.maps && (it.maps.group_a_url || it.maps.contrast_url) && (
              <div className="maps-stack">
                {it.maps.group_a_url && (
                  <figure className="map-fig">
                    <figcaption>{it.maps.group_a_label} — mean TRIBE v2 activation (fsaverage5)</figcaption>
                    <img className="surface-map" src={`${API}${it.maps.group_a_url}`} alt="" />
                  </figure>
                )}
                {it.maps.group_b_url && (
                  <figure className="map-fig">
                    <figcaption>{it.maps.group_b_label} — mean TRIBE v2 activation (fsaverage5)</figcaption>
                    <img className="surface-map" src={`${API}${it.maps.group_b_url}`} alt="" />
                  </figure>
                )}
                {it.maps.contrast_url && (
                  <figure className="map-fig">
                    <figcaption>Contrast: {condA} − {condB} (red = {condA}&gt;{condB}, blue = {condB}&gt;{condA})</figcaption>
                    <img className="surface-map" src={`${API}${it.maps.contrast_url}`} alt="" />
                  </figure>
                )}
              </div>
            )}

            {roiMeta?.destrieux_parcels && (
              <div className="roi-meta">
                {exp.focal_roi} = mean over {roiMeta.n_vertices} vertices · Destrieux parcel{' '}
                <code>{roiMeta.destrieux_parcels.join(', ')}</code>
              </div>
            )}

            {/* Per-clip datapoints */}
            <div className="clip-grid">
              {[...itDpA, ...itDpB].map((d, i) => (
                <div key={i} className="clip-cell" title={d.caption || ''}>
                  {d.url ? <video src={`${API}${d.url}`} muted loop autoPlay playsInline className="clip-vid" />
                    : <div className="stim-placeholder">clip</div>}
                  <span className={`trial-cond ${d.condition === condA ? 'cond-a' : 'cond-b'}`}>{d.condition}</span>
                  <span className="clip-val">{d.value.toFixed(3)}</span>
                </div>
              ))}
            </div>

            {it.stats && it.stats.p_value != null && <StatsStrip stats={it.stats} a={condA} b={condB} />}
            {it.decision && (
              <div className="decision-box">
                <span className={`decision-tag ${it.decision.status}`}>{it.decision.status}</span>
                <span className="decision-text">{it.decision.stop_reason}</span>
              </div>
            )}
            {it.refinement?.diagnosis && (
              <div className="refine-box">
                <span className="refine-tag">refine</span>
                <span className="refine-text">{it.refinement.diagnosis}</span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function StatsStrip({ stats, a, b }: { stats: any; a: string; b: string }) {
  return (
    <div className="stats-strip">
      <div className="stat-pill"><span className="sp-label">{a}</span><span className="sp-val">{stats.mean_a?.toFixed(3)}</span></div>
      <div className="stat-pill"><span className="sp-label">{b}</span><span className="sp-val">{stats.mean_b?.toFixed(3)}</span></div>
      <div className="stat-pill"><span className="sp-label">p</span><span className={`sp-val ${stats.significant ? 'sig' : ''}`}>{stats.p_value?.toFixed(3)}</span></div>
      <div className="stat-pill"><span className="sp-label">Cohen's d</span><span className="sp-val">{stats.cohens_d?.toFixed(2)} <em>({stats.effect_size_label})</em></span></div>
    </div>
  )
}
