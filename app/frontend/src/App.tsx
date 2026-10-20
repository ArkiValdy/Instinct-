import { useEffect, useRef, useState } from "react"
import { analyzeThumbnail } from "./api"

function Speedometer({ score }: { score: number }) {
  const cx = 130,
    cy = 115,
    r = 85

  const toRad = (deg: number) => (deg * Math.PI) / 180

  const polarToCart = (deg: number, radius = r) => ({
    x: cx + radius * Math.cos(toRad(deg)),
    y: cy - radius * Math.sin(toRad(deg)),
  })

  const arc = (startDeg: number, endDeg: number, radius = r) => {
    const s = polarToCart(startDeg, radius)
    const e = polarToCart(endDeg, radius)
    const span = startDeg - endDeg
    const large = span > 180 ? 1 : 0
    return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`
  }

  const clamped = Math.max(0, Math.min(100, score))
  const scoreAngle = 210 - (clamped / 100) * 240
  const needle = polarToCart(scoreAngle, 62)

  const color =
    clamped <= 33 ? "#22c55e" : clamped <= 66 ? "#f59e0b" : "#ef4444"
  const label = clamped <= 33 ? "Low" : clamped <= 66 ? "Medium" : "High"

  const ticks = [0, 25, 50, 75, 100]

  return (
    <svg width="260" height="190" viewBox="0 0 260 190">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Background track */}
      <path
        d={arc(210, -30)}
        fill="none"
        stroke="#222"
        strokeWidth="16"
        strokeLinecap="round"
      />

      {/* Coloured zones — low, mid, high */}
      <path
        d={arc(210, 130)}
        fill="none"
        stroke="#22c55e"
        strokeWidth="16"
        strokeLinecap="butt"
        opacity="0.18"
      />
      <path
        d={arc(130, 30)}
        fill="none"
        stroke="#f59e0b"
        strokeWidth="16"
        strokeLinecap="butt"
        opacity="0.18"
      />
      <path
        d={arc(30, -30)}
        fill="none"
        stroke="#ef4444"
        strokeWidth="16"
        strokeLinecap="butt"
        opacity="0.18"
      />

      {/* Progress track */}
      {clamped > 0 && (
        <path
          d={arc(210, scoreAngle)}
          fill="none"
          stroke={color}
          strokeWidth="16"
          strokeLinecap="round"
          filter="url(#glow)"
        />
      )}

      {/* Tick marks */}
      {ticks.map((pct) => {
        const a = 210 - (pct / 100) * 240
        const inner = polarToCart(a, r - 24)
        const outer = polarToCart(a, r - 10)
        return (
          <line
            key={pct}
            x1={inner.x}
            y1={inner.y}
            x2={outer.x}
            y2={outer.y}
            stroke="#3a3a3a"
            strokeWidth="2"
            strokeLinecap="round"
          />
        )
      })}

      {/* Needle shadow */}
      <line
        x1={cx}
        y1={cy}
        x2={needle.x}
        y2={needle.y}
        stroke="rgba(0,0,0,0.6)"
        strokeWidth="5"
        strokeLinecap="round"
      />
      {/* Needle */}
      <line
        x1={cx}
        y1={cy}
        x2={needle.x}
        y2={needle.y}
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        filter="url(#glow)"
      />

      {/* Center cap */}
      <circle
        cx={cx}
        cy={cy}
        r="9"
        fill="#1a1a1a"
        stroke={color}
        strokeWidth="2"
      />
      <circle cx={cx} cy={cy} r="4" fill={color} />

      {/* Score number */}
      <text
        x={cx}
        y={cy + 32}
        textAnchor="middle"
        fill="white"
        fontSize="36"
        fontWeight="500"
        fontFamily="DM Mono, monospace"
      >
        {clamped}
      </text>
      <text
        x={cx}
        y={cy + 52}
        textAnchor="middle"
        fill={color}
        fontSize="11"
        fontFamily="Outfit, sans-serif"
        fontWeight="600"
        letterSpacing="3"
      >
        {label.toUpperCase()}
      </text>
    </svg>
  )
}

const GENRE_LABELS: Record<string, string> = {
  genre1: "Genre 1",
}

export default function App() {
  const [thumbnail, setThumbnail] = useState<string | null>(null)
  const [thumbnailFile, setThumbnailFile] = useState<File | null>(null)
  const [title, setTitle] = useState("")
  const [genre, setGenre] = useState("genre1")
  const [showResults, setShowResults] = useState(false)
  const [score, setScore] = useState(0)
  const [feedback, setFeedback] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const requestController = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      requestController.current?.abort()
      if (thumbnail) URL.revokeObjectURL(thumbnail)
    }
  }, [thumbnail])

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setThumbnailFile(file)
      setThumbnail((current) => {
        if (current) URL.revokeObjectURL(current)
        return URL.createObjectURL(file)
      })
      setShowResults(false)
      setError(null)
    }
  }

  const handleAnalyze = async () => {
    if (!thumbnailFile || !title.trim() || isAnalyzing) return

    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller

    setIsAnalyzing(true)
    setError(null)
    setShowResults(false)

    try {
      const result = await analyzeThumbnail(
        {
          thumbnail: thumbnailFile,
          title: title.trim(),
          genre,
        },
        controller.signal,
      )

      setScore(result.score)
      setFeedback(result.feedback)
      setShowResults(true)
    } catch (requestError) {
      if (
        requestError instanceof DOMException &&
        requestError.name === "AbortError"
      ) {
        return
      }

      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to analyze the thumbnail. Please try again.",
      )
    } finally {
      if (requestController.current === controller) {
        requestController.current = null
        setIsAnalyzing(false)
      }
    }
  }

  const canAnalyze = thumbnailFile !== null && title.trim() !== ""

  return (
    <div
      className="min-h-screen bg-[#080808] text-white flex flex-col items-center py-12 px-6"
      style={{ fontFamily: "'Outfit', sans-serif" }}
    >
      {/* YouTube red top accent bar */}
      <div className="fixed top-0 left-0 right-0 h-[3px] bg-[#FF0000] z-50" />

      {/* Fixed AI Powered badge — bottom right */}
      <div className="fixed bottom-5 right-5 z-50">
        <span className="inline-block text-[#FF0000] text-[10px] font-mono tracking-[0.22em] uppercase border border-[#FF0000]/30 bg-[#080808]/90 backdrop-blur px-3 py-1.5 rounded">
          AI Powered
        </span>
      </div>
      {/* Header */}
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold tracking-tight">
          Instict<span className="text-[#FF0000]">+</span>
        </h1>
        <p className="text-[#555] text-sm mt-2 max-w-xs mx-auto">
          Upload your thumbnail, enter a title, and get an instant clickbait
          score.
        </p>
      </header>

      {/* Inputs */}
      <div className="w-full max-w-xl space-y-5">
        {/* Thumbnail upload */}
        <div>
          <label className="block text-[10px] text-[#666] uppercase tracking-[0.18em] mb-2 font-medium">
            Video Thumbnail
          </label>
          <div
            onClick={() => fileRef.current?.click()}
            className="relative border border-[#242424] hover:border-[#FF0000]/40 rounded-xl overflow-hidden h-40 flex flex-col items-center justify-center cursor-pointer transition-all duration-200 bg-[#0f0f0f] group"
          >
            {thumbnail ? (
              <>
                <img
                  src={thumbnail}
                  alt="Uploaded thumbnail"
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="text-xs text-white font-medium tracking-wide">
                    Change image
                  </span>
                </div>
              </>
            ) : (
              <>
                <svg
                  className="w-9 h-9 text-[#333] group-hover:text-[#FF0000]/60 mb-3 transition-colors"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                <span className="text-[#444] text-sm group-hover:text-[#666] transition-colors">
                  Click to upload
                </span>
                <span className="text-[#333] text-xs mt-1">PNG, JPG, WEBP</span>
              </>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />
        </div>

        {/* Title */}
        <div>
          <label className="block text-[10px] text-[#666] uppercase tracking-[0.18em] mb-2 font-medium">
            Video Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => {
              setTitle(e.target.value)
              setShowResults(false)
              setError(null)
            }}
            placeholder="e.g. I Survived 100 Days in the Jungle..."
            className="w-full bg-[#0f0f0f] border border-[#242424] focus:border-[#FF0000]/50 rounded-xl px-4 py-3 text-white placeholder-[#383838] outline-none transition-colors text-sm"
          />
        </div>

        {/* Genre */}
        <div>
          <label className="block text-[10px] text-[#666] uppercase tracking-[0.18em] mb-2 font-medium">
            Genre
          </label>
          <div className="relative">
            <select
              value={genre}
              onChange={(e) => {
                setGenre(e.target.value)
                setShowResults(false)
                setError(null)
              }}
              className="w-full bg-[#0f0f0f] border border-[#242424] focus:border-[#FF0000]/50 rounded-xl px-4 py-3 text-white outline-none transition-colors text-sm appearance-none cursor-pointer"
            >
              <option value="genre1">Genre 1</option>
            </select>
            <svg
              className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#444] pointer-events-none"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </div>
        </div>

        {/* Analyze button */}
        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze || isAnalyzing}
          className="w-full py-3.5 rounded-xl font-semibold text-sm tracking-wider transition-all duration-200 bg-[#FF0000] text-white hover:bg-[#e60000] active:scale-[0.99] disabled:opacity-20 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? "Analyzing…" : "Analyze Thumbnail"}
        </button>

        {error && (
          <div
            role="alert"
            className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300"
          >
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {showResults && (
        <div className="w-full max-w-4xl mt-14">
          <div className="border-t border-[#161616] pt-10">
            {/* Centered inputs summary */}
            <div className="flex flex-col items-center mb-10">
              <div className="rounded-xl overflow-hidden border border-[#222] shadow-xl shadow-black/50 mb-5">
                <img
                  src={thumbnail!}
                  alt={title}
                  className="w-72 h-40 object-cover block"
                />
              </div>
              <h2 className="text-lg font-semibold text-center max-w-sm leading-snug">
                {title}
              </h2>
              <span className="mt-3 text-[10px] text-[#888] uppercase tracking-[0.2em] border border-[#272727] rounded-md px-3 py-1.5">
                {GENRE_LABELS[genre] ?? genre}
              </span>
            </div>

            {/* Feedback + Speedometer */}
            <div className="flex flex-col md:flex-row gap-4 items-stretch">
              {/* Feedback */}
              <div className="flex-1 bg-[#0f0f0f] border border-[#1c1c1c] rounded-2xl p-7">
                <p className="text-[10px] text-[#555] uppercase tracking-[0.2em] mb-4 font-medium">
                  Feedback
                </p>
                <p className="text-[#bbb] text-sm leading-relaxed whitespace-pre-line">
                  {feedback}
                </p>
              </div>

              {/* Speedometer */}
              <div className="bg-[#0f0f0f] border border-[#1c1c1c] rounded-2xl p-7 flex flex-col items-center justify-center md:min-w-[300px]">
                <p className="text-[10px] text-[#555] uppercase tracking-[0.2em] mb-1 font-medium">
                  Clickbait Score
                </p>
                <Speedometer score={score} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
