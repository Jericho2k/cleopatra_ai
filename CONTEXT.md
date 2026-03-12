# Cleopatra AI — AI Reply Suggestion System for OFM Agencies

## What Cleopatra does
Suggests 3 AI-generated reply options to chatters managing
OnlyFans creator accounts. AI mimics the specific creator's
voice and adapts to conversation stage.

## Stack
- Python 3.11 + FastAPI
- Supabase (Postgres + pgvector)
- Together AI — Llama 3.1 70B (uncensored)
- Upstash Redis
- Next.js frontend on Vercel

## Strict rules for code generation
- One file per request, never touch other files
- All Pydantic models → models/schemas.py only
- All DB queries → db/queries.py only
- All env vars → core/config.py only
- No new dependencies without explicit approval
- No logic in main.py, routes call functions only
- Never restructure or rename existing files

## Project structure
cleopatra/
├── main.py
├── core/
│   ├── config.py
│   └── supabase.py
├── models/
│   └── schemas.py
├── db/
│   └── queries.py
├── services/
│   └── suggestions.py
├── ai/
│   ├── generator.py
│   ├── prompt_builder.py
│   ├── stage_classifier.py
│   └── rag.py
└── persona/
    └── extractor.py