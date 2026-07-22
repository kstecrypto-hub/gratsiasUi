# Yeastar Call Analyzer frontend

The administrator interface is a standard Next.js application. It uses only the authenticated backend API; it contains no demo records, provider mocks, or runtime fixtures.

## Development

Use Node.js 22 or later.

```bash
npm ci
npm run dev
```

Browser requests to `/api/*` are proxied to `BACKEND_URL`, which defaults to `http://backend:8000` for Docker Compose. Set `BACKEND_URL=http://localhost:8000` when running the frontend directly on the host.

## Verification

```bash
npm run lint
npm run build
npx playwright install chromium
npm test
```

Playwright uses isolated request interception in `tests/e2e`; production modules never import test code.
