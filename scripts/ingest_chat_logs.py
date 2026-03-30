"""
Fansly Chat Log Ingestion Script
=================================
Parses raw Fansly API JSON responses (single object or list of chunks),
extracts clean fan/creator exchange pairs, embeds them with OpenAI,
and stores in Supabase pgvector for RAG.

Usage:
    # Dry run first to verify output:
    python ingest_chat_logs.py --creator_id <uuid> --account_id <fansly_id> --files *.json --dry_run

    # Then run for real:
    python ingest_chat_logs.py --creator_id <uuid> --account_id <fansly_id> --files *.json
"""

import json
import os
import re
import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
from dotenv import load_dotenv
import openai
from supabase import create_client

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

EMBEDDING_MODEL = "text-embedding-3-small"
TABLE = "chat_log_embeddings"
SIMILARITY_THRESHOLD = 0.85
MIN_CHARS = 8
FAN_GROUP_GAP = 120  # seconds


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def is_spam(text: str) -> bool:
    patterns = [
        r"stay subscribed with renew on",
        r"hot gift in dms every 15th",
        r"upgrade to my next sub tier",
        r"fans\.ly/subscriptions",
        r"enjoy my erotic content",
        r"cum at least twice a day",
        r"hi sweetie.*let.*make a deal",
        r"tell me about yourself.*little gift",
        r"p\.s\. stay subscribed",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def parse_file(filepath: str, creator_account_id: str) -> list[dict]:
    """Parse Fansly API format or synthetic augmented format."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle synthetic format from augment_chat_logs.py
    if isinstance(data, dict) and data.get("synthetic"):
        msgs = []
        for i, pair in enumerate(data.get("pairs", [])):
            msgs.append({"role": "fan", "content": pair["fan"], "ts": i*2, "id": f"syn_fan_{i}"})
            msgs.append({"role": "creator", "content": pair["creator"], "ts": i*2+1, "id": f"syn_creator_{i}"})
        return msgs

    chunks = data if isinstance(data, list) else [data]

    all_msgs = []
    for chunk in chunks:
        messages = chunk.get("response", {}).get("messages", [])
        for msg in messages:
            content = (msg.get("content") or "").strip()
            if not content or len(content) < MIN_CHARS:
                continue
            if is_spam(content):
                continue
            role = "creator" if msg["senderId"] == creator_account_id else "fan"
            all_msgs.append({
                "role": role,
                "content": content,
                "ts": msg.get("createdAt", 0),
                "id": msg.get("id"),
            })

    all_msgs.sort(key=lambda m: m["ts"])

    seen_ids = set()
    unique = []
    for msg in all_msgs:
        if msg["id"] not in seen_ids:
            seen_ids.add(msg["id"])
            unique.append(msg)

    return unique


def group_fan_turns(messages: list[dict]) -> list[dict]:
    """Merge consecutive fan messages within FAN_GROUP_GAP seconds into one turn."""
    grouped = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "fan":
            group = [msg["content"]]
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "fan":
                if messages[j]["ts"] - messages[j-1]["ts"] <= FAN_GROUP_GAP:
                    group.append(messages[j]["content"])
                    j += 1
                else:
                    break
            grouped.append({"role": "fan", "content": " ".join(group), "ts": msg["ts"]})
            i = j
        else:
            grouped.append(msg)
            i += 1
    return grouped


def extract_exchanges(messages: list[dict]) -> list[tuple[str, str]]:
    """Extract (fan_message, creator_reply) pairs, filtering duplicate creator replies."""
    pairs = []
    recent_creator = []
    for i in range(len(messages) - 1):
        cur, nxt = messages[i], messages[i + 1]
        if cur["role"] == "fan" and nxt["role"] == "creator":
            reply = nxt["content"]
            is_dup = any(
                similarity(reply, prev) > SIMILARITY_THRESHOLD
                for prev in recent_creator[-5:]
            )
            if not is_dup:
                pairs.append((cur["content"], reply))
                recent_creator.append(reply)
    return pairs


def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def ingest_pairs(pairs: list[tuple[str, str]], creator_id: str, dry_run: bool = False):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Ingesting {len(pairs)} exchange pairs...")
    inserted = 0
    skipped = 0

    for i, (fan_msg, creator_reply) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] Fan: {fan_msg[:70]}")
        print(f"         Creator: {creator_reply[:70]}")

        if dry_run:
            continue

        try:
            embedding = get_embedding(fan_msg)
            supabase.table(TABLE).insert({
                "creator_id": creator_id,
                "fan_message": fan_msg,
                "creator_reply": creator_reply,
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": "real",
            }).execute()
            inserted += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            skipped += 1

    if not dry_run:
        print(f"\nDone. Inserted: {inserted}, Skipped: {skipped}")
    else:
        print(f"\n[DRY RUN] Would insert {len(pairs)} pairs. Run without --dry_run to ingest.")


def main():
    parser = argparse.ArgumentParser(description="Ingest Fansly chat logs into Cleopatra AI")
    parser.add_argument("--creator_id", required=True, help="Cleopatra AI creator UUID")
    parser.add_argument("--account_id", required=True, help="Fansly creator account ID")
    parser.add_argument("--files", nargs="+", required=True, help="JSON files to ingest")
    parser.add_argument("--dry_run", action="store_true", help="Preview without inserting")
    args = parser.parse_args()

    all_pairs = []

    for filepath in args.files:
        print(f"\nProcessing: {filepath}")
        messages = parse_file(filepath, args.account_id)
        print(f"  Raw messages after filtering: {len(messages)}")
        messages = group_fan_turns(messages)
        print(f"  After grouping fan turns: {len(messages)}")
        pairs = extract_exchanges(messages)
        print(f"  Exchange pairs: {len(pairs)}")
        all_pairs.extend(pairs)

    # Global deduplication across files
    unique_pairs = []
    for pair in all_pairs:
        is_dup = any(
            similarity(pair[0], ex[0]) > SIMILARITY_THRESHOLD and
            similarity(pair[1], ex[1]) > SIMILARITY_THRESHOLD
            for ex in unique_pairs
        )
        if not is_dup:
            unique_pairs.append(pair)

    print(f"\nTotal unique pairs after deduplication: {len(unique_pairs)}")
    ingest_pairs(unique_pairs, args.creator_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()