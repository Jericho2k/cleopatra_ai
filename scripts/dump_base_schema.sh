#!/usr/bin/env bash
#
# Produce the authoritative base schema (DB-000).
#
# The base definitions of creators, fans, messages, suggestions,
# chatter_creators, scheduled_actions, the vault tables, the fan list tables and
# the fan_conversation_summaries view exist only in the live Supabase project.
# Until they are in version control, their primary keys, foreign keys, unique
# constraints, cascade behaviour, nullability and indexes cannot be reviewed,
# reproduced, or tested — which is what blocks the index questions in
# db/MIGRATIONS.md and made REL-002's uniqueness situation unverifiable.
#
# This dumps SCHEMA ONLY. No rows, no owners, no privileges, no credentials.
#
# Usage:
#   SUPABASE_DB_URL='postgresql://postgres:...@db.<ref>.supabase.co:5432/postgres' \
#       scripts/dump_base_schema.sh
#
# Then review the diff before committing. In particular check that it contains
# no data, no role passwords, and nothing from auth/storage/vault.
set -euo pipefail

: "${SUPABASE_DB_URL:?set SUPABASE_DB_URL to a read-capable connection string}"

OUT="$(dirname "$0")/../db/000_base_schema.sql"

# --schema-only  : no rows, ever.
# --no-owner     : no ALTER ... OWNER TO lines tied to the Supabase roles.
# --no-privileges: no GRANTs; the migrations manage those.
# --schema=public: application objects only. Supabase's managed auth, storage,
#                  realtime, vault and extensions schemas are deliberately
#                  excluded — they are platform-owned and do not belong in an
#                  application migration baseline.
pg_dump \
    --schema-only \
    --no-owner \
    --no-privileges \
    --schema=public \
    --file="$OUT" \
    "$SUPABASE_DB_URL"

echo "Wrote $OUT"
echo
echo "Before committing:"
echo "  1. grep -i 'INSERT INTO\\|COPY ' \"$OUT\"   # must be empty"
echo "  2. grep -i 'password\\|secret\\|key ' \"$OUT\" # must contain no credentials"
echo "  3. Read db/MIGRATIONS.md § Switching CI to the real base schema"
