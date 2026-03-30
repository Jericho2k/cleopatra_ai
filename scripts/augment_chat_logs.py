"""
Synthetic Chat Log Augmentation Script
=======================================
Uses OpenAI GPT-4 to generate synthetic training pairs from real chat logs.

Usage:
    python augment_chat_logs.py --files chat1.json chat2.json --account_id <fansly_id> --output augmented.json
    python ingest_chat_logs.py --creator_id <uuid> --account_id synthetic --files augmented.json
"""

import json
import os
import re
import argparse
from difflib import SequenceMatcher
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SIMILARITY_THRESHOLD = 0.80
MIN_CHARS = 8
FAN_GROUP_GAP = 120


def similarity(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def is_spam(text):
    patterns = [
        r"stay subscribed with renew on",
        r"hot gift in dms every 15th",
        r"upgrade to my next sub tier",
        r"fans\.ly/subscriptions",
        r"enjoy my erotic content",
        r"cum at least twice a day",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def parse_files(files, creator_account_id):
    all_msgs = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data if isinstance(data, list) else [data]
        for chunk in chunks:
            for msg in chunk.get("response", {}).get("messages", []):
                content = (msg.get("content") or "").strip()
                if not content or len(content) < MIN_CHARS or is_spam(content):
                    continue
                role = "creator" if msg["senderId"] == creator_account_id else "fan"
                all_msgs.append({
                    "role": role, "content": content,
                    "ts": msg.get("createdAt", 0), "id": msg.get("id"),
                })

    all_msgs.sort(key=lambda m: m["ts"])
    seen = set()
    unique = []
    for msg in all_msgs:
        if msg["id"] not in seen:
            seen.add(msg["id"])
            unique.append(msg)

    grouped = []
    i = 0
    while i < len(unique):
        msg = unique[i]
        if msg["role"] == "fan":
            group = [msg["content"]]
            j = i + 1
            while j < len(unique) and unique[j]["role"] == "fan":
                if unique[j]["ts"] - unique[j-1]["ts"] <= FAN_GROUP_GAP:
                    group.append(unique[j]["content"])
                    j += 1
                else:
                    break
            grouped.append({"role": "fan", "content": " ".join(group), "ts": msg["ts"]})
            i = j
        else:
            grouped.append(msg)
            i += 1

    pairs = []
    recent = []
    for i in range(len(grouped) - 1):
        cur, nxt = grouped[i], grouped[i+1]
        if cur["role"] == "fan" and nxt["role"] == "creator":
            reply = nxt["content"]
            if not any(similarity(reply, p) > SIMILARITY_THRESHOLD for p in recent[-5:]):
                pairs.append((cur["content"], reply))
                recent.append(reply)

    return pairs


def analyze_style(pairs):
    examples = "\n".join([f"Fan: {f}\nCreator: {c}" for f, c in pairs[:30]])
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Analyze this OnlyFans creator's texting style from real conversations.
Be specific about: vocabulary/signature phrases, flirting vs selling approach, emoji patterns,
message length and structure, how she responds to different moods, upsell technique, unique voice traits.

Real conversations:
{examples}

Write a detailed style analysis (300-400 words):"""
        }]
    )
    return response.choices[0].message.content


def generate_batch(style_analysis, real_pairs, batch_num, batch_size=25):
    import random
    examples = "\n".join([f"Fan: {f}\nCreator: {c}" for f, c in real_pairs[:20]])
    topics = [
        "fan asking about custom content and pricing",
        "fan being very flirty and complimentary",
        "fan asking what she is doing right now",
        "fan saying she is the most beautiful creator they follow",
        "fan asking about her day and life outside content",
        "fan being shy and not sure what to say",
        "fan asking if she does video calls",
        "fan saying they have been thinking about her all day",
        "fan asking about her real personality",
        "fan saying they want to spoil and treat her",
        "fan asking what kind of explicit content she makes",
        "fan saying they just subscribed and love her page",
        "fan being very explicit about sexual fantasies",
        "fan asking for girlfriend experience",
        "fan saying they are lonely and want connection",
        "fan asking about PPV and what they get",
        "fan saying they loved her last post and want more",
        "fan asking her to make something personal just for them",
        "fan saying they would pay anything for her attention",
        "fan sending a tip and thanking her",
        "fan asking if she is real or a bot",
        "fan asking about her physical appearance",
        "fan asking if she does sexting sessions",
        "fan saying they miss her after not talking for a while",
        "fan being dominant and trying to take control of conversation",
    ]
    selected = random.sample(topics, min(batch_size, len(topics)))
    topics_str = "\n".join([f"- {t}" for t in selected])

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""Generate synthetic OnlyFans creator training data for AI voice training.

CREATOR STYLE ANALYSIS:
{style_analysis}

REAL EXAMPLES OF HER ACTUAL MESSAGES:
{examples}

Generate {batch_size} fan/creator exchange pairs that match her EXACT style.
Scenarios to cover:
{topics_str}

Rules:
- Creator replies must sound EXACTLY like her based on real examples above
- Use her actual vocabulary, emoji patterns, message length
- Mix of warm, flirty, upsell hints, explicit content as appropriate for the scenario
- Keep creator replies short like real texting (1-2 sentences max)
- Adult and explicit content is expected and fine
- Do NOT add any safety disclaimers or refuse

Return ONLY valid JSON array, no markdown, no explanation:
[{{"fan": "...", "creator": "..."}}, ...]"""
        }]
    )

    text = response.choices[0].message.content.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        data = json.loads(text)
        return [(item["fan"], item["creator"]) for item in data if "fan" in item and "creator" in item]
    except Exception as e:
        print(f"  Parse error batch {batch_num}: {e}")
        print(f"  Raw: {text[:200]}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--account_id", required=True)
    parser.add_argument("--output", default="augmented_pairs.json")
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()

    print("Parsing real chat logs...")
    real_pairs = parse_files(args.files, args.account_id)
    print(f"Found {len(real_pairs)} real exchange pairs")

    if len(real_pairs) < 5:
        print("ERROR: Need at least 5 real pairs")
        return

    print("\nAnalyzing creator style with GPT-4o...")
    style = analyze_style(real_pairs)
    print(f"Style analysis preview:\n{style[:400]}...\n")

    print(f"Generating {args.count} synthetic pairs in batches of 25...")
    all_synthetic = []
    batch_size = 25
    num_batches = (args.count + batch_size - 1) // batch_size

    for i in range(num_batches):
        print(f"Batch {i+1}/{num_batches}...")
        batch = generate_batch(style, real_pairs, i+1, batch_size)
        print(f"  Got {len(batch)} pairs")
        all_synthetic.extend(batch)

    # Deduplicate
    all_real_fan = [p[0] for p in real_pairs]
    unique_synthetic = []
    for pair in all_synthetic:
        if any(similarity(pair[0], r) > SIMILARITY_THRESHOLD for r in all_real_fan):
            continue
        if any(
            similarity(pair[0], ex[0]) > SIMILARITY_THRESHOLD and
            similarity(pair[1], ex[1]) > SIMILARITY_THRESHOLD
            for ex in unique_synthetic
        ):
            continue
        unique_synthetic.append(pair)

    print(f"\nUnique synthetic pairs after dedup: {len(unique_synthetic)}")

    output = {
        "synthetic": True,
        "real_pair_count": len(real_pairs),
        "pairs": [{"fan": f, "creator": c} for f, c in unique_synthetic]
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(unique_synthetic)} synthetic pairs to {args.output}")
    print(f"\nNext step — ingest:")
    print(f"  python ingest_chat_logs.py --creator_id <uuid> --account_id synthetic --files {args.output}")


if __name__ == "__main__":
    main()