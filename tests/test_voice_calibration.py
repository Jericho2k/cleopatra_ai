import asyncio
from types import SimpleNamespace

from ai.voice_calibration import render_voice_calibration, voice_style_summary
from models.schemas import Persona
from services import voice_calibration
from services.voice_calibration import normalize_message_ids


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        return self

    def select(self, _fields):
        return self

    def eq(self, _field, _value):
        return self

    def in_(self, _field, _values):
        return self

    def order(self, _field, **_kwargs):
        return self

    def limit(self, _value):
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


def test_voice_calibration_is_off_by_default():
    persona = Persona(voice_calibration_samples=["honestly idk lol"])
    assert render_voice_calibration(persona) == ""


def test_voice_calibration_renders_only_approved_style_evidence():
    persona = Persona(
        voice_calibration_enabled=True,
        voice_calibration_samples=[
            "honestly idk lol",
            "wait what 😭",
            "no I haven't seen it",
        ],
    )
    rendered = render_voice_calibration(persona)

    assert "CREATOR VOICE CALIBRATION — BETA" in rendered
    assert "operator-approved examples" in rendered
    assert "<voice_sample" in rendered
    assert "never conversation facts" in rendered
    assert "Never reuse its fan-specific details" in rendered


def test_voice_sample_normalization_is_bounded_and_deduplicated():
    persona = Persona(
        voice_calibration_samples=[
            "  honestly   idk  ",
            "HONESTLY IDK",
            "another one",
        ]
    )
    assert persona.voice_calibration_samples == ["honestly idk", "another one"]

    ids = normalize_message_ids(["a", "a", "", "b", *range(40)])
    assert ids[:2] == ["a", "b"]
    assert len(ids) == 30


def test_voice_style_summary_describes_observed_distribution():
    summary = voice_style_summary(
        [
            "honestly idk",
            "wait what 😭",
            "are you serious?",
            "okay",
        ]
    )
    assert "median" in summary
    assert "lowercase starts" in summary
    assert "contain a question" in summary
    assert "observed emoji" in summary


def test_candidate_discovery_never_auto_approves_known_ai_messages(monkeypatch):
    rows = [
        {"id": "human", "content": "  honestly idk  ", "was_ai_suggested": False},
        {"id": "unknown", "content": "wait what", "was_ai_suggested": None},
        {"id": "ai", "content": "polished generated reply", "was_ai_suggested": True},
    ]
    monkeypatch.setattr(voice_calibration, "get_supabase", lambda: _Query(rows))

    candidates = asyncio.run(
        voice_calibration.list_voice_calibration_candidates("creator-1")
    )

    assert [row["id"] for row in candidates] == ["human", "unknown"]
    assert all("approved" not in row for row in candidates)


def test_save_requires_explicit_ids_and_deduplicates_sample_text(monkeypatch):
    rows = [
        {"id": "a", "content": "same message", "was_ai_suggested": False},
        {"id": "b", "content": "Same message", "was_ai_suggested": False},
        {"id": "ai", "content": "generated", "was_ai_suggested": True},
    ]
    saved: list[Persona] = []

    async def _get_persona(_creator_id):
        return Persona()

    async def _save_persona(_creator_id, persona):
        saved.append(persona)

    monkeypatch.setattr(voice_calibration, "get_supabase", lambda: _Query(rows))
    monkeypatch.setattr(voice_calibration, "get_creator_persona", _get_persona)
    monkeypatch.setattr(voice_calibration, "save_persona", _save_persona)

    persona = asyncio.run(
        voice_calibration.save_voice_calibration(
            "creator-1",
            enabled=True,
            approved_message_ids=["a", "b", "ai"],
        )
    )

    assert persona.voice_calibration_enabled is True
    assert persona.voice_calibration_message_ids == ["a"]
    assert persona.voice_calibration_samples == ["same message"]
    assert saved == [persona]
