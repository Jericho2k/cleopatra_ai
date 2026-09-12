"""A recoverable planning failure must never become "Full Auto sent nothing".

The production trace this pins down::

    action=CREATE_PAID_SESSION
    [SESSION] plan-session status=selected_set_unavailable
    [SIMULATION] outcome=no_send

The same fan message, retried seconds later, planned a valid $30 set. Nothing
about that turn was a decision to stay silent: the commercial layer had decided
to sell, and a stale package snapshot in the middle of the pipeline turned that
decision into silence — reported to the operator as the product working.

Two invariants are tested here, and they pull in opposite directions, which is
exactly why both need pinning:

* a recoverable failure is repaired and the turn continues;
* a package the fan ACCEPTED by name and price is never silently swapped for a
  different one, however good the replacement is. He is shown it instead.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.commercial import (
    CreatorPolicy,
    FanCommercialState,
    FanStatus,
    PackageOption,
)
from services import session_plan_recovery
from services.session_plan_recovery import (
    PlanFailureClass,
    classify_plan_status,
    invalidates_accepted_contract,
    is_recoverable,
    recover_session_plan,
)


def run(coro):
    return asyncio.run(coro)


def package(package_id: str, price_cents: int, *set_ids: str) -> PackageOption:
    return PackageOption(
        package_id=package_id,
        label="quick private session",
        price_cents=price_cents,
        set_id=set_ids[0] if set_ids else None,
        set_ids=list(set_ids),
        asset_types=["photo_set" for _ in set_ids],
        step_count=len(set_ids) or 1,
    )


# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "selected_set_unavailable",
        "no_coherent_sequence",
        "no_valid_allocation",
        "missing_confirmed_budget",
    ],
)
def test_stale_plan_states_are_recoverable(status):
    assert classify_plan_status(status) is PlanFailureClass.RECOVERABLE
    assert is_recoverable(status) is True


def test_an_empty_vault_is_an_impossible_sale_not_a_bug():
    assert classify_plan_status("no_sets") is PlanFailureClass.IMPOSSIBLE
    assert is_recoverable("no_sets") is False


def test_ok_is_not_a_failure_at_all():
    assert classify_plan_status("ok") is PlanFailureClass.INTENTIONAL
    assert classify_plan_status(None) is PlanFailureClass.INTENTIONAL


def test_an_unknown_status_fails_toward_recovery():
    """Recovery still continues the turn and reports honestly; treating an
    unrecognised status as impossible would reintroduce the silence."""
    assert classify_plan_status("some_future_status") is PlanFailureClass.RECOVERABLE


def test_only_contract_invalidating_statuses_re_present():
    assert invalidates_accepted_contract("selected_set_unavailable") is True
    assert invalidates_accepted_contract("no_valid_allocation") is True
    assert invalidates_accepted_contract("no_coherent_sequence") is False


# ---------------------------------------------------------------------------
# 2. Recovery
# ---------------------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch):
    """Stub the four collaborators recovery uses, and record every state write."""
    state = FanCommercialState(status=FanStatus.OFFER_SELECTED)
    state.selected_package_id = "package:quick:gone"
    state.selected_package_set_ids = ["gone"]
    state.selected_package_set_id = "gone"
    state.selected_package_price_cents = 3000
    state.confirmed_budget_cents = 3000

    store: dict = {
        "state": state,
        "saved": [],
        "packages": [package("package:quick:fresh", 3000, "fresh-1", "fresh-2")],
        "plan_results": [{"status": "ok", "session": {"status": "active", "plan": [{}]}}],
        "plan_calls": [],
    }

    async def fake_get_fan_state(_fan_id):
        return store["state"]

    async def fake_get_creator_policy(_creator_id):
        return CreatorPolicy()

    async def fake_save_fan_state(_fan_id, _creator_id, saved_state):
        store["saved"].append(
            {
                "status": saved_state.status,
                "selected_package_id": saved_state.selected_package_id,
                "selected_package_set_ids": list(saved_state.selected_package_set_ids),
                "confirmed_budget_cents": saved_state.confirmed_budget_cents,
                "offered_packages": list(saved_state.offered_packages),
            }
        )

    async def fake_get_packages(*_args, **_kwargs):
        return list(store["packages"]), ("photo_set",)

    def fake_select(packages, _price_learning, max_options=2):
        return list(packages)[:max_options]

    async def fake_plan(_creator_id, _fan_id, **kwargs):
        store["plan_calls"].append(kwargs)
        return store["plan_results"].pop(0)

    monkeypatch.setattr("db.commercial_queries.get_fan_state", fake_get_fan_state)
    monkeypatch.setattr("db.commercial_queries.get_creator_policy", fake_get_creator_policy)
    monkeypatch.setattr("db.commercial_queries.save_fan_state", fake_save_fan_state)
    monkeypatch.setattr(
        "db.commercial_queries.get_offerable_packages_with_inventory", fake_get_packages
    )
    monkeypatch.setattr("services.price_learning.select_recommended_packages", fake_select)
    monkeypatch.setattr("services.session_planner.plan_session_for_fan", fake_plan)
    return store


def test_a_generic_stale_plan_is_replanned_and_the_turn_continues(wired):
    """No accepted contract: recomputing from current inventory is the repair."""
    recovery = run(
        recover_session_plan(
            creator_id="creator-1",
            fan_id="fan-1",
            status="selected_set_unavailable",
            had_accepted_contract=False,
        )
    )

    assert recovery.recovered is True
    assert recovery.turn_may_continue is True
    assert recovery.present_replacement is False
    assert recovery.session is not None
    # It replanned against the freshly computed package, not the stale snapshot.
    assert wired["plan_calls"][0]["selected_set_ids"] == ["fresh-1", "fresh-2"]


def test_the_stale_selection_is_cleared_before_anything_else_reads_it(wired):
    run(
        recover_session_plan(
            creator_id="creator-1",
            fan_id="fan-1",
            status="selected_set_unavailable",
            had_accepted_contract=False,
        )
    )
    # Whatever else happened, the dead package id is gone from persisted state.
    assert all(
        saved["selected_package_id"] != "package:quick:gone"
        for saved in wired["saved"]
    )


def test_an_accepted_package_is_re_presented_never_substituted(wired):
    """He said yes to a specific thing at a specific price. A different thing at
    that price is a substitution he never agreed to."""
    recovery = run(
        recover_session_plan(
            creator_id="creator-1",
            fan_id="fan-1",
            status="selected_set_unavailable",
            had_accepted_contract=True,
        )
    )

    assert recovery.recovered is False, "no silent replacement session"
    assert recovery.present_replacement is True
    assert recovery.accepted_contract_lost is True
    assert recovery.turn_may_continue is True
    assert [p.package_id for p in recovery.replacement_packages] == [
        "package:quick:fresh"
    ]
    # Nothing was planned, and the accepted budget no longer stands.
    assert wired["plan_calls"] == []
    assert wired["saved"][-1]["status"] is FanStatus.OFFER_PENDING
    assert wired["saved"][-1]["confirmed_budget_cents"] is None


def test_no_replacement_inventory_continues_the_conversation_without_an_offer(wired):
    wired["packages"] = []

    recovery = run(
        recover_session_plan(
            creator_id="creator-1",
            fan_id="fan-1",
            status="selected_set_unavailable",
            had_accepted_contract=True,
        )
    )

    assert recovery.recovered is False
    assert recovery.continue_without_offer is True
    assert recovery.turn_may_continue is True, (
        "a fan who cannot be sold to is still a fan who gets a reply"
    )
    assert wired["saved"][-1]["status"] is FanStatus.IDLE


def test_recovery_attempts_replanning_exactly_once(wired):
    """A retry loop would burn the turn instead of answering the fan."""
    wired["plan_results"] = [{"status": "selected_set_unavailable", "session": None}]

    recovery = run(
        recover_session_plan(
            creator_id="creator-1",
            fan_id="fan-1",
            status="no_coherent_sequence",
            had_accepted_contract=False,
        )
    )

    assert len(wired["plan_calls"]) == 1
    assert recovery.recovered is False
    assert recovery.present_replacement is True
    assert recovery.turn_may_continue is True


def test_an_impossible_sale_is_not_repaired(wired):
    recovery = run(
        recover_session_plan(
            creator_id="creator-1",
            fan_id="fan-1",
            status="no_sets",
            had_accepted_contract=False,
        )
    )
    assert recovery.failure_class is PlanFailureClass.IMPOSSIBLE
    assert recovery.continue_without_offer is True
    assert wired["plan_calls"] == []


# ---------------------------------------------------------------------------
# 3. The call site
# ---------------------------------------------------------------------------


def test_the_auto_path_no_longer_returns_on_a_failed_plan():
    """The exact line that produced outcome=no_send is gone.

    Asserted against the source because the surrounding function needs the whole
    Full Auto pipeline to exercise, and the invariant is structural: a non-ok
    plan status must reach recovery, not a bare return.
    """
    source = (
        Path(__file__).resolve().parents[1] / "services" / "suggestions.py"
    ).read_text(encoding="utf-8")

    assert "_recover_or_downgrade_plan(" in source
    assert "AUTO_OUTCOME_PLAN_UNRECOVERABLE" in source
    # The recovery branch is what a failed plan reaches under Commercial v2.
    assert 'print(f"[SESSION] plan-session status={plan_data.get(\'status\')} fan={fan_id}")' not in source
    assert "recovery_outcome" in source


def test_an_unrecoverable_plan_reports_a_distinct_simulator_outcome():
    from services.suggestions import (
        AUTO_OUTCOME_NO_SEND,
        AUTO_OUTCOME_PLAN_UNRECOVERABLE,
    )

    assert AUTO_OUTCOME_PLAN_UNRECOVERABLE != AUTO_OUTCOME_NO_SEND
    assert AUTO_OUTCOME_PLAN_UNRECOVERABLE == "plan_unrecoverable"


def test_every_planner_status_is_classified():
    """A status the planner can return but this module has never heard of is a
    silent gap, so the two lists are compared directly."""
    planner = (
        Path(__file__).resolve().parents[1] / "services" / "session_planner.py"
    ).read_text(encoding="utf-8")

    import re

    # Only the failure returns: ``{"status": ..., "session": None}``. The
    # session dict the planner BUILDS also carries a status, and "active" is
    # not a planner outcome.
    returned = set(
        re.findall(r'"status": "([a-z_]+)",\s*\n?\s*"session": None', planner)
    )
    known = set(session_plan_recovery._CLASSIFICATION)
    assert returned <= known, f"unclassified planner statuses: {returned - known}"
