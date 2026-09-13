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


def test_a_one_candidate_turn_keeps_only_the_first_reply():
    """Full Auto under writer_v3 asks for one reply and accepts one.

    A model that ignores the instruction and returns three alternatives must not
    be able to put two unused phrasings in front of any downstream chooser.
    """
    payload = json.dumps(["the real reply", "an alternative", "a third"])
    replies = parse_reply_candidates(
        payload, Persona(avg_message_length="short"), max_candidates=1
    )
    assert replies == ["the real reply"]


def test_a_one_candidate_turn_still_skips_a_rejected_first_reply():
    """The cap is on how many are KEPT, not on how many are looked at."""
    payload = json.dumps(["as an ai i can help", "what i actually want to say"])
    replies = parse_reply_candidates(payload, Persona(), max_candidates=1)
    assert replies == ["what i actually want to say"]


def test_the_default_cardinality_is_unchanged():
    payload = json.dumps(["one", "two", "three", "four"])
    replies = parse_reply_candidates(payload, Persona())
    assert len(replies) == 3
