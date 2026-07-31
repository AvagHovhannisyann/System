# Frontend — Operator Dashboard

Next.js 15 (App Router) frontend for the systematic equity research platform.
TypeScript strict, Tailwind CSS v4, shadcn/ui, TanStack Query. Functional UI
only — visual design is delegated (see `DECISIONS.md` D-006).

## Layout

- `app/` — routes. Only Overview (`/`) is routed today; other dashboard
  sections render as inert nav entries until their backend phase lands.
- `components/` — `providers.tsx` (QueryClientProvider), `sidebar-nav.tsx`,
  `ui/` (stock shadcn/ui components).
- `lib/api.ts` — typed backend API client (`fetchHealth`).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL, inlined at build time |

## Commands

```bash
npm install        # install dependencies
npm run dev        # dev server on http://localhost:3000
npm run lint       # eslint
npx tsc --noEmit   # typecheck
npm run build      # production build (standalone output for Docker)
```
