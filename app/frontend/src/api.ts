export type AnalysisRequest = {
  thumbnail: File
  title: string
  genre: string
}

export type AnalysisResponse = {
  score: number
  feedback: string
  analysisId?: string
}

const ANALYSIS_API_URL =
  import.meta.env.VITE_ANALYSIS_API_URL?.trim() || "/api/analyze"

function isAnalysisResponse(value: unknown): value is AnalysisResponse {
  if (typeof value !== "object" || value === null) return false

  const response = value as Record<string, unknown>
  return (
    typeof response.score === "number" &&
    Number.isFinite(response.score) &&
    response.score >= 0 &&
    response.score <= 100 &&
    typeof response.feedback === "string" &&
    response.feedback.trim().length > 0 &&
    (response.analysisId === undefined ||
      typeof response.analysisId === "string")
  )
}

export async function analyzeThumbnail(
  request: AnalysisRequest,
  signal?: AbortSignal,
): Promise<AnalysisResponse> {
  const body = new FormData()
  body.append("thumbnail", request.thumbnail)
  body.append("title", request.title)
  body.append("genre", request.genre)

  const response = await fetch(ANALYSIS_API_URL, {
    method: "POST",
    body,
    signal,
    headers: {
      Accept: "application/json",
    },
  })

  if (!response.ok) {
    let message = `Analysis failed (${response.status})`

    try {
      const errorBody = (await response.json()) as { message?: unknown }
      if (typeof errorBody.message === "string" && errorBody.message.trim()) {
        message = errorBody.message
      }
    } catch {
      // The status code is still useful when the backend does not return JSON.
    }

    throw new Error(message)
  }

  const result: unknown = await response.json()
  if (!isAnalysisResponse(result)) {
    throw new Error("The analysis service returned an invalid response")
  }

  return result
}
