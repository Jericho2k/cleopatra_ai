"""The Experience Director: one scene, read as a table of turns.

Every test here is a sequence of fan turns run through the pure
``advance_scene``. The whole point of keeping the transition function pure is
that the choreography can be asserted as choreography — no database, no
writer, no model — so a regression shows up as "the beat is wrong", not as a
prompt that reads slightly differently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.experience_director import (  # noqa: E402
    ExperienceBeat,
    FanReaction,
    SceneState,
    advance_scene,
    classify_fan_reaction,
    scene_from_row,
)
from services.experience_director import scene_allows_new_offer  # noqa: E402


SHOWER_SET = {
    "scene_key": "shower:none",
    "premise": "photos in the shower",
    "reveals": "undressed in the shower",
    "continuation": "what happens once she is",
    "intensity_level": 4,
}


def _unlock(previous=None, metadata=None):
    return advance_scene(
        previous=previous,
        commercial_decision={
            "_unlock_confirmed": True,
            "accepted_offer_set_id": "shower-1",
        },
        scene_metadata=metadata or SHOWER_SET,
    )


# --- the core rule: a purchase hands the next move to him --------------------


def test_a_confirmed_purchase_goes_to_await_reaction_not_back_to_selling():
    scene = _unlock()
    assert scene.beat is ExperienceBeat.AWAIT_REACTION
    assert scene.another_unlock_ready is False
    assert scene.reaction_processed is False
    assert scene.last_unlocked_set_id == "shower-1"
    assert scene.unlocks_in_scene == 1
    assert scene_allows_new_offer(scene) is False


def test_his_first_message_after_the_unlock_is_the_reaction_and_is_owed_an_answer():
    scene = advance_scene(previous=_unlock(), latest_fan_message="holy shit that was hot")
    assert scene.beat is ExperienceBeat.PLAY
    assert scene.last_fan_reaction is FanReaction.POSITIVE
    assert scene.reaction_processed is False, "this very reply owes him that reaction"
    assert scene.writer_context()["reaction_owed"] is True
    assert scene_allows_new_offer(scene) is False, "play first, sell later"


def test_the_scene_stays_in_play_while_he_is_still_in_it():
    """No offer discovery just because turns have passed.

    This is the difference between the Experience Director and the fixed
    cooldown it replaced: time does not open the gate, dialogue does.
    """
    scene = advance_scene(previous=_unlock(), latest_fan_message="damn")
    for message in ("yeah", "ok", "mm"):
        scene = advance_scene(previous=scene, latest_fan_message=message)
        assert scene.beat is ExperienceBeat.PLAY
        assert scene_allows_new_offer(scene) is False


# --- what DOES open the gate -------------------------------------------------


def test_asking_for_more_opens_the_gate_immediately():
    """He accelerates his own progression. No message count is consulted."""
    scene = advance_scene(previous=_unlock(), latest_fan_message="what else do you have")
    assert scene.beat is ExperienceBeat.BRIDGE
    assert scene.another_unlock_ready is True
    assert scene.transition_reason == "fan_asked_for_more"
    assert scene_allows_new_offer(scene) is True


def test_asking_for_more_works_on_the_very_first_post_unlock_turn():
    scene = advance_scene(previous=_unlock(), latest_fan_message="send more")
    assert scene.beat is ExperienceBeat.BRIDGE
    assert scene_allows_new_offer(scene) is True


def test_a_hot_scene_that_keeps_climbing_produces_a_bridge():
    """The ordinary route: the conversation itself earns the next moment."""
    scene = _unlock()
    scene = advance_scene(
        previous=scene,
        latest_fan_message="fuck that was so good",
        situation={"wants_explicit": "true", "conversation_energy": "rising"},
    )
    assert scene.beat is ExperienceBeat.PLAY
    scene = advance_scene(
        previous=scene,
        latest_fan_message="god I keep thinking about it",
        situation={"wants_explicit": "true", "conversation_energy": "rising"},
        scene_metadata=SHOWER_SET,
    )
    assert scene.beat is ExperienceBeat.BRIDGE
    assert scene_allows_new_offer(scene) is True


def test_a_flat_reaction_never_produces_a_bridge_however_long_he_talks():
    scene = advance_scene(previous=_unlock(), latest_fan_message="ok cool")
    for _ in range(6):
        scene = advance_scene(
            previous=scene,
            latest_fan_message="yeah",
            situation={"conversation_energy": "flat"},
        )
    assert scene.beat is ExperienceBeat.PLAY
    assert scene_allows_new_offer(scene) is False


def test_a_bad_reaction_is_dealt_with_rather_than_sold_past():
    scene = advance_scene(
        previous=_unlock(),
        latest_fan_message="honestly that was pretty disappointing",
    )
    assert scene.last_fan_reaction is FanReaction.NEGATIVE
    assert scene.another_unlock_ready is False
    assert scene_allows_new_offer(scene) is False
    assert scene.writer_context()["fan_reaction"] == "NEGATIVE"


# --- the first offer was never gated -----------------------------------------


def test_a_scene_with_no_unlock_yet_never_blocks_the_first_offer():
    scene = advance_scene(
        situation={"wants_explicit": "true", "conversation_energy": "rising"}
    )
    assert scene.unlocks_in_scene == 0
    assert scene_allows_new_offer(scene) is True
    assert scene_allows_new_offer(None) is True
    assert scene_allows_new_offer({}) is True


# --- safety and closure ------------------------------------------------------


def test_a_handoff_closes_the_scene():
    scene = advance_scene(
        previous=_unlock(),
        commercial_decision={"action": "HAND_OFF_TO_HUMAN"},
    )
    assert scene.beat is ExperienceBeat.CLOSE
    assert scene.another_unlock_ready is False


def test_an_affordability_pause_closes_the_scene():
    scene = advance_scene(
        previous=_unlock(),
        commercial_decision={"action": "PAUSE_NO_BUDGET"},
    )
    assert scene.beat is ExperienceBeat.CLOSE
    assert scene_allows_new_offer(scene) is False


# --- what the writer is allowed to see ---------------------------------------


def test_the_writer_projection_carries_no_money_and_no_sequence():
    scene = advance_scene(previous=_unlock(), latest_fan_message="that was amazing")
    context = scene.writer_context()

    forbidden = {
        "price",
        "price_cents",
        "last_unlocked_set_id",
        "another_unlock_ready",
        "unlocks_in_scene",
        "beats_in_scene",
        "turns_since_unlock",
        "total",
        "step",
    }
    assert forbidden.isdisjoint(context), (
        "the writer projection must not carry a price, a set id, a counter or "
        "anything that reveals a sequence exists"
    )
    serialised = str(context).lower()
    assert "$" not in serialised
    assert "cents" not in serialised


def test_the_scene_survives_a_round_trip_through_its_stored_row():
    scene = advance_scene(previous=_unlock(), latest_fan_message="so hot")
    restored = scene_from_row(scene.to_context())
    assert restored.beat is scene.beat
    assert restored.last_fan_reaction is scene.last_fan_reaction
    assert restored.last_unlocked_set_id == scene.last_unlocked_set_id


def test_an_unreadable_stored_scene_starts_over_rather_than_raising():
    """Choreography must never be able to stop a reply going out."""
    assert scene_from_row({"beat": "NOT_A_BEAT"}).beat is ExperienceBeat.SETUP

    junk = scene_from_row({"intimacy_level": "banana"})
    fresh = SceneState()
    assert junk.beat is fresh.beat
    assert junk.intimacy_level == fresh.intimacy_level
    assert junk.another_unlock_ready is fresh.another_unlock_ready


# --- reaction reading --------------------------------------------------------


def test_reaction_classification_reads_his_words_not_the_analyzer():
    assert classify_fan_reaction("holy fuck") is FanReaction.POSITIVE
    assert classify_fan_reaction("send me more") is FanReaction.WANTS_MORE
    assert classify_fan_reaction("that was a waste") is FanReaction.NEGATIVE
    assert classify_fan_reaction("ok") is FanReaction.NEUTRAL
    assert classify_fan_reaction("") is FanReaction.NONE


def test_a_direction_named_before_the_purchase_does_not_open_the_gate_after_it():
    """The bridge has to be produced by the dialogue, not by a stored field.

    ``desired_direction`` is sticky for the whole scene — it is what the writer
    is told he has been steering towards. If it also counted as a bridge, then
    every fan who ever named a theme would become offerable on the first
    processed reaction, which is a stored value opening the gate rather than
    the conversation doing it.
    """
    before = advance_scene(
        situation={"desired_experience": "shower", "wants_explicit": "true"},
        latest_fan_message="i want the shower one",
    )
    assert before.desired_direction == "shower"

    scene = _unlock(previous=before.to_context())
    scene = advance_scene(previous=scene.to_context(), latest_fan_message="nice")
    scene = advance_scene(previous=scene.to_context(), latest_fan_message="cool")

    assert scene.desired_direction == "shower", "still carried for the writer"
    assert scene.beat is ExperienceBeat.PLAY
    assert scene_allows_new_offer(scene) is False


def test_naming_a_new_direction_after_the_purchase_does_open_it():
    """The control: the same field, said on THIS turn, is a bridge."""
    scene = _unlock()
    scene = advance_scene(previous=scene.to_context(), latest_fan_message="that was hot")
    scene = advance_scene(
        previous=scene.to_context(),
        situation={"desired_experience": "the bedroom ones"},
        latest_fan_message="what about the bedroom ones",
    )
    assert scene.beat is ExperienceBeat.BRIDGE
    assert scene_allows_new_offer(scene) is True
