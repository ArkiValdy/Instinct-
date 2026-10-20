# Instinct+ frontend

React/Vite frontend for submitting a video thumbnail, title, and genre to an analysis backend.

## Run locally

Requirements: Node.js 22 and pnpm 10.

```bash
pnpm install
copy .env.example .env.local
pnpm dev
```

Set `VITE_ANALYSIS_API_URL` in `.env.local` to the backend analysis endpoint. It defaults to `/api/analyze`, which is convenient when the frontend and backend are served from the same origin or a development proxy is used.

## Backend API contract

The frontend sends `POST VITE_ANALYSIS_API_URL` as `multipart/form-data`. The browser sets the multipart boundary automatically, so the frontend deliberately does not set a `Content-Type` header.

Request fields:

| Field | Type | Description |
| --- | --- | --- |
| `thumbnail` | File | Uploaded PNG, JPEG, WEBP, or other browser-accepted image |
| `title` | string | Trimmed video title |
| `genre` | string | Selected genre identifier, currently `genre1` |

Successful response (`2xx`):

```json
{
  "score": 72,
  "feedback": "Specific analysis text returned by the backend.",
  "analysisId": "optional-backend-id"
}
```

`score` must be a finite number from 0 through 100 and `feedback` must be a non-empty string. `analysisId` is optional. The frontend validates the response before rendering it.

Error response (`4xx` or `5xx`):

```json
{
  "message": "A user-facing explanation of the failure."
}
```

If the API runs on another origin, it must allow the frontend origin through CORS. Authentication headers can be added in `src/api.ts` once the backend's authentication scheme is known.

## Commands

- `pnpm dev` — start Vite in development mode
- `pnpm build` — type-check and build the production bundle
- `pnpm format` — format the source
- `pnpm typecheck` — check TypeScript without building
