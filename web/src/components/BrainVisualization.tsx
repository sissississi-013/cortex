import { useMemo } from 'react'

interface Region {
  region_id: string
  label: string
  hemisphere: string
  mni_coordinates: number[]
  activation_intensity: number
  activation_level: string
  displayValue?: number
}

interface Props {
  regions: Region[]
  title?: string
  focalRoi?: string
}

const REGION_POSITIONS: Record<string, { x: number; y: number; side: 'left' | 'right' | 'center' }> = {
  V1: { x: 200, y: 260, side: 'center' },
  V2: { x: 215, y: 250, side: 'center' },
  A1: { x: 90, y: 160, side: 'left' },
  STG: { x: 75, y: 175, side: 'left' },
  STS: { x: 80, y: 195, side: 'left' },
  IFG: { x: 100, y: 110, side: 'left' },
  MTG: { x: 70, y: 210, side: 'left' },
  AG: { x: 110, y: 230, side: 'left' },
  SMG: { x: 95, y: 215, side: 'left' },
  FFA: { x: 290, y: 230, side: 'right' },
  PPA: { x: 155, y: 240, side: 'left' },
  dlPFC: { x: 115, y: 85, side: 'left' },
  mPFC: { x: 200, y: 65, side: 'center' },
  ACC: { x: 200, y: 100, side: 'center' },
  PCC: { x: 200, y: 210, side: 'center' },
  amygdala: { x: 155, y: 195, side: 'left' },
  hippocampus: { x: 160, y: 210, side: 'left' },
  insula: { x: 120, y: 145, side: 'left' },
  M1: { x: 150, y: 60, side: 'left' },
  cerebellum: { x: 240, y: 280, side: 'right' },
  thalamus: { x: 190, y: 165, side: 'center' },
  precuneus: { x: 200, y: 185, side: 'center' },
  TPJ: { x: 85, y: 230, side: 'left' },
  OFC: { x: 160, y: 100, side: 'left' },
  Wernicke: { x: 70, y: 195, side: 'left' },
}

function intensityColor(intensity: number): string {
  if (intensity > 0.8) return '#f4fff3'
  if (intensity > 0.6) return '#95e57f'
  if (intensity > 0.4) return '#10d96b'
  return '#718071'
}

function intensityRadius(intensity: number): number {
  return 6 + intensity * 14
}

export function BrainVisualization({ regions, title, focalRoi }: Props) {
  const regionMap = useMemo(() => {
    const map = new Map<string, Region>()
    for (const r of regions) {
      map.set(r.region_id, r)
    }
    return map
  }, [regions])

  return (
    <div className="brain-viz">
      {title && <h3 className="brain-viz-title">{title}</h3>}
      <svg viewBox="0 0 400 340" className="brain-svg">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <ellipse cx="200" cy="170" rx="170" ry="155" fill="rgba(149, 229, 127, 0.06)" stroke="rgba(149, 229, 127, 0.22)" strokeWidth="1" />

        <path d="M200 15 Q200 170 200 325" stroke="rgba(149, 229, 127, 0.18)" strokeWidth="1" strokeDasharray="4 4" />

        <text x="60" y="30" fill="#718071" fontSize="10" fontFamily="monospace">L</text>
        <text x="330" y="30" fill="#718071" fontSize="10" fontFamily="monospace">R</text>
        <text x="190" y="30" fill="#718071" fontSize="10" fontFamily="monospace">A</text>
        <text x="190" y="335" fill="#718071" fontSize="10" fontFamily="monospace">P</text>

        {Object.entries(REGION_POSITIONS).map(([id, pos]) => {
          const region = regionMap.get(id)
          if (!region) {
            return (
              <circle
                key={id}
                cx={pos.x}
                cy={pos.y}
                r={3}
                fill="rgba(149, 229, 127, 0.08)"
                stroke="rgba(149, 229, 127, 0.18)"
                strokeWidth="0.5"
              />
            )
          }

          const r = intensityRadius(region.activation_intensity)
          const color = intensityColor(region.activation_intensity)
          const isFocal = focalRoi === id

          return (
            <g key={id} filter="url(#glow)">
              {isFocal && (
                <circle cx={pos.x} cy={pos.y} r={r + 5} fill="none"
                  stroke="#95e57f" strokeWidth="1.5" strokeDasharray="3 2" opacity={0.9} />
              )}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={r}
                fill={color}
                opacity={0.15 + region.activation_intensity * 0.5}
              />
              <circle
                cx={pos.x}
                cy={pos.y}
                r={r * 0.6}
                fill={color}
                opacity={0.4 + region.activation_intensity * 0.4}
              />
              <text
                x={pos.x}
                y={pos.y - r - 3}
                fill={isFocal ? "#f4fff3" : "#b9cabb"}
                fontSize="8"
                fontWeight={isFocal ? 700 : 400}
                textAnchor="middle"
                fontFamily="monospace"
              >
                {id}
              </text>
            </g>
          )
        })}
      </svg>
      {regions.length > 0 && (
        <div className="brain-legend">
          {regions.slice(0, 5).map((r) => (
            <div key={r.region_id} className="legend-item">
              <span
                className="legend-dot"
                style={{ background: intensityColor(r.activation_intensity) }}
              />
              <span className="legend-label" style={focalRoi === r.region_id ? { color: '#f4fff3', fontWeight: 600 } : undefined}>
                {r.label}{focalRoi === r.region_id ? ' (focal)' : ''}
              </span>
              <span className="legend-value">
                {(r.displayValue ?? r.activation_intensity).toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
