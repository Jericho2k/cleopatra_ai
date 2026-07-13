import json

from ai.generator import parse_reply_candidates
from models.schemas import Persona


def test_parse_reply_candidates_accepts_three_short_replies():
    payload = json.dumps(["come closer", "tell me what you want", "maybe i have an idea"])
    replies = parse_reply_candidates(payload, Persona(avg_message_length="short"))
    assert len(replies) == 3


def test_parse_reply_candidates_fails_closed_on_non_json():
    replies = parse_reply_candidates("not json", Persona())
    assert replies == []


def test_parse_reply_candidates_rejects_robotic_candidates():
    payload = json.dumps(["certainly", "of course", "as an ai"])
    replies = parse_reply_candidates(payload, Persona())
    assert replies == []
