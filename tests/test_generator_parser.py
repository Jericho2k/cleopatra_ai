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


def test_short_average_is_not_a_hard_word_limit_for_a_substantive_answer():
    from ai.generator import parse_auto_messages_outcome

    reply = ('You mentioned two different problems with the download. First, tell me '
             'whether the attachment opens at all. Then we can check the missing sound '
             'without asking you to buy the same item again.')
    assert len(reply.split()) > 25
    persona = Persona(avg_message_length='short')
    assert parse_reply_candidates(json.dumps([reply]), persona) == [reply]
    assert parse_auto_messages_outcome(json.dumps({'messages': [reply]}), persona).replies == [reply]
    # Several bubbles form one reply, and may legitimately exceed 25 words too.
    result = parse_auto_messages_outcome(json.dumps({'messages': [reply, 'Which one happened first?']}), persona)
    assert result.replies == [reply + ' | Which one happened first?']
