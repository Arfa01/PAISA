# Security maintenance

This log records dependency-security maintenance without reproducing exploit
code. It does not replace the repository's normal test and review requirements.

## 24 October 2026 — Zod constrained-collection advisory

| Item | Recorded value |
|---|---|
| Advisory | `GHSA-mrhx-c7xc-m3w5` / `CVE-2023-54404` |
| Affected dependency | Direct frontend production dependency `zod` |
| Previous version | Declared `^3.23.8`; lockfile resolved `3.25.76` |
| Patched version | Exact `4.6.7` |
| Migration date | 2026-10-24 |

The controlled Zod 4 migration preserved the dashboard's response contracts,
made record key schemas explicit, and added the backend's documented 1,000-row
table-page and 5,000-row historical-price-page bounds to the runtime validators.
As defense in depth, every unpaginated API array and open evidence record also
has a 1,000-entry frontend safety ceiling. Oversized collections are rejected
before their elements or record values are validated, preventing malformed
responses from amplifying validation work and issue allocation. The ceiling
comfortably preserves the unified six-model response and the current evidence
metadata; it does not replace the backend's endpoint contracts. This work did
not upgrade another dependency or add fallback data.

Verification commands:

```bash
cd frontend
npm ci
npm run test
npm run lint
npm run build
npm audit --omit=dev
npm ls zod --all
cd ..
python -m pytest
python -m pip check
python -m compileall src scripts
git diff --check
```
