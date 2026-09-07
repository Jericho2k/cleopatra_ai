# Deployment environment and fail-closed configuration

Audit reference: **SEC-004**, **SEC-005**.

## `APP_ENV` is required

`APP_ENV` selects the deployment mode. It is resolved in exactly one place,
`core/environment.py`, and every security gate reads that resolution.

| `APP_ENV` value | Resolves to | Effect |
|---|---|---|
| `development`, `dev`, `local` | development | Relaxed local auth; `/test/*` routes available |
| `test`, `testing` | test | Production auth enforcement; `/test/*` routes available |
| `production`, `prod` | production | Everything enforced; `/test/*` routes return 404 |
| unset, empty, or anything else (`staging`, `Prod `, a typo) | **production** | Fail closed, with a warning on stdout |

The important property: **the absence of the variable selects the safe mode.**
Before this change the default was `development`, so one missing Railway variable
turned off dashboard authentication, every tenancy check, and webhook signature
verification at the same time.

`test` is deliberately not a development mode. The backend test-suite asserts the
production enforcement paths, so it must keep failing closed; what `test` unlocks
is the local helper routes and nothing security-relevant.

## What fails closed when `APP_ENV` is not development

- `require_dashboard` / `api_auth_middleware` — an unconfigured
  `DASHBOARD_API_SECRET` returns 500 rather than allowing the request through.
- `authenticated_dashboard_user` — a request with no bearer token is rejected
  with 401 rather than returning `None`.
- `require_creator_access` — a `None` operator raises instead of returning
  early, so tenancy is never a no-op.
- `fansly_webhook` — a missing `WEBHOOK_SECRET` returns 500 rather than skipping
  signature verification.
- `/test/simulate-ppv-purchase` and `/test/inject-message` — return 404.

## Local development

Set it explicitly. This is the intended, documented break:

```bash
export APP_ENV=development
```

Without it a local server now behaves like production and will refuse requests
until `DASHBOARD_API_SECRET` and `WEBHOOK_SECRET` are configured.

## Deployment checklist

1. Confirm `APP_ENV=production` is set in the Railway service variables. Do not
   rely on the default — there is no permissive default any more, but an
   explicit value keeps the startup log unambiguous.
2. Confirm `DASHBOARD_API_SECRET` and `WEBHOOK_SECRET` are set. With
   `APP_ENV=production` an unset secret is a hard 500, not a silent bypass.
3. After deploy, check the startup line:

   ```
   [STARTUP] APP_ENV=production resolved=production
   ```

   A line reading `resolved=production` with `APP_ENV=<unset>` means the variable
   is missing — the deploy is safe but misconfigured, and the `/test/*` routes
   and relaxed auth are correctly off.
