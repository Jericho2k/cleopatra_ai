"""What to do when a commercial plan cannot be produced for a turn.

Production evidence this exists for::

    action=CREATE_PAID_SESSION
    [SESSION] plan-session status=selected_set_unavailable
    [SIMULATION] outcome=no_send

The same fan message, retried seconds later, planned a valid $30 set. Nothing
about that turn was an intentional decision to stay silent — the commercial
layer had decided to sell, and a stale snapshot in the middle of the pipeline
turned that decision into silence. "Full Auto decided to send nothing" is a
product statement, and it must never be printed over a recoverable bug.

The four classes
----------------
``RECOVERABLE``
    The commercial decision is still right; the *plan* referred to content that
    is no longer sellable (already sent, un-approved, re-sliced by a vault
    resync) or to a total that cannot be split across the steps that remain.
    Recompute from current approved unsent inventory and continue the turn.

``IMPOSSIBLE``
    There is genuinely nothing to sell this fan right now. Not a bug: the turn
    continues as ordinary conversation, without an offer.

``INFRASTRUCTURE``
    The planner raised. Nothing is known about inventory, so nothing is claimed.

``INTENTIONAL``
    Policy chose not to sell. Never reaches this module; named so the four
    outcomes the audit asked for are all enumerated in one place.

The contract boundary
---------------------
Recovery may silently replace content the fan has not yet agreed to. It may NOT
silently replace a package he accepted by name and price: that is a different
product at the same price, and he never said yes to it. When the accepted
contract is unavailable the turn downgrades to presenting the valid replacement,
so the fan is told and asked, and the commercial state stops claiming a
selection that cannot be delivered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlanFailureClass(str, Enum):
    RECOVERABLE = "recoverable"
    IMPOSSIBLE = "impossible"
    INFRASTRUCTURE = "infrastructure"
    INTENTIONAL = "intentional"


# Statuses services/session_planner.py can return, and what each one means.
#
# selected_set_unavailable — the snapshot the fan's state points at no longer
#   matches sellable inventory. Stale state, not a dead sale.
# no_coherent_sequence — nothing coherent at the requested shape/budget. The
#   inventory may still support a different, honestly presented package.
# no_valid_allocation — the accepted total cannot be split across these steps
#   without inventing a price. A different step count usually can be.
# missing_confirmed_budget — the planner was called without a contract. That is
#   a caller bug, and recomputing options is exactly the repair.
# no_sets — there is genuinely nothing approved and unsent left for this fan.
_CLASSIFICATION: dict[str, PlanFailureClass] = {
    "selected_set_unavailable": PlanFailureClass.RECOVERABLE,
    "no_coherent_sequence": PlanFailureClass.RECOVERABLE,
    "no_valid_allocation": PlanFailureClass.RECOVERABLE,
    "missing_confirmed_budget": PlanFailureClass.RECOVERABLE,
    "no_sets": PlanFailureClass.IMPOSSIBLE,
}

# Statuses whose cause is a stale *accepted contract* rather than a stale
# generic plan. These are the ones that must be re-presented, never substituted.
CONTRACT_INVALIDATING_STATUSES = frozenset(
    {"selected_set_unavailable", "no_valid_allocation"}
)

RECOVERABLE_STATUSES = frozenset(
    status
    for status, kind in _CLASSIFICATION.items()
    if kind is PlanFailureClass.RECOVERABLE
)


def classify_plan_status(status: str | None) -> PlanFailureClass:
    """Classify one ``plan_session_for_fan`` status. Unknown means recoverable.

    Failing toward recovery is the safe direction: a recovery attempt that finds
    nothing still continues the turn and reports an honest outcome, whereas
    treating an unrecognised status as impossible reintroduces the silence this
    module exists to remove.
    """
    if not status or status == "ok":
        return PlanFailureClass.INTENTIONAL
    return _CLASSIFICATION.get(str(status), PlanFailureClass.RECOVERABLE)


def is_recoverable(status: str | None) -> bool:
    return classify_plan_status(status) is PlanFailureClass.RECOVERABLE


def invalidates_accepted_contract(status: str | None) -> bool:
    """True when the exact thing the fan accepted can no longer be delivered."""
    return str(status or "") in CONTRACT_INVALIDATING_STATUSES


@dataclass(frozen=True)
class PlanRecovery:
    """The outcome of one recovery attempt.

    ``session`` is set when a valid plan now exists and the turn may proceed as
    the commercial decision intended. ``present_replacement`` is set when a valid
    replacement exists but the fan must be shown it rather than charged for it.
    ``continue_without_offer`` is set when the sale is genuinely not available and
    the turn should continue as conversation — never as silence.
    """

    status: str
    failure_class: PlanFailureClass
    session: dict | None = None
    replacement_packages: list = None  # list[PackageOption]
    present_replacement: bool = False
    continue_without_offer: bool = False
    accepted_contract_lost: bool = False
    reason: str = ""

    @property
    def recovered(self) -> bool:
        return self.session is not None

    @property
    def turn_may_continue(self) -> bool:
        return bool(
            self.session
            or self.present_replacement
            or self.continue_without_offer
        )


async def recover_session_plan(
    *,
    creator_id: str,
    fan_id: str,
    status: str,
    had_accepted_contract: bool,
    desired_experience: str | None = None,
    price_learning: dict | None = None,
    hard_ceiling_cents: int | None = None,
) -> PlanRecovery:
    """Repair a recoverable planning failure, or say honestly that it cannot.

    The order is deliberate. Stale selection state is cleared FIRST, because
    leaving it in place is what makes the next turn fail the same way; then
    options are recomputed from current approved unsent inventory; only then is
    a replanning attempt made, and only when the fan had not already accepted an
    exact contract.
    """
    from db.commercial_queries import (
        get_creator_policy,
        get_fan_state,
        get_offerable_packages_with_inventory,
        save_fan_state,
    )
    from models.commercial import FanStatus
    from services.price_learning import select_recommended_packages
    from services.session_planner import plan_session_for_fan

    failure_class = classify_plan_status(status)
    if failure_class is not PlanFailureClass.RECOVERABLE:
        return PlanRecovery(
            status=status,
            failure_class=failure_class,
            continue_without_offer=failure_class is PlanFailureClass.IMPOSSIBLE,
            reason="not_recoverable",
        )

    state = await get_fan_state(fan_id)
    policy = await get_creator_policy(creator_id)

    # Clear the stale snapshot before anything else reads it. A selection that
    # cannot be delivered is not a selection.
    state.selected_package_id = None
    state.selected_package_set_id = None
    state.selected_package_set_ids = []
    state.selected_package_label = None
    state.selected_package_price_cents = None
    if state.status == FanStatus.OFFER_SELECTED:
        state.status = FanStatus.OFFER_PENDING

    packages, _vault_types = await get_offerable_packages_with_inventory(
        creator_id,
        fan_id,
        policy,
        price_learning=price_learning or {},
        desired_experience=desired_experience or None,
        hard_ceiling_cents=hard_ceiling_cents,
    )
    packages = select_recommended_packages(
        packages,
        price_learning or {},
        max_options=2 if policy.offer_two_packages else 1,
    )

    if not packages:
        state.offered_packages = []
        state.status = FanStatus.IDLE
        state.confirmed_budget_cents = None
        await save_fan_state(fan_id, creator_id, state)
        return PlanRecovery(
            status=status,
            failure_class=failure_class,
            continue_without_offer=True,
            accepted_contract_lost=had_accepted_contract,
            reason="no_replacement_inventory",
        )

    state.offered_packages = packages

    if had_accepted_contract and invalidates_accepted_contract(status):
        # He said yes to a specific thing at a specific price. Delivering a
        # different thing at that price — even a better one — is a substitution
        # he never agreed to. Present it and let him choose again.
        state.status = FanStatus.OFFER_PENDING
        state.confirmed_budget_cents = None
        await save_fan_state(fan_id, creator_id, state)
        return PlanRecovery(
            status=status,
            failure_class=failure_class,
            replacement_packages=packages,
            present_replacement=True,
            accepted_contract_lost=True,
            reason="accepted_contract_unavailable",
        )

    # Nothing was accepted yet (or the failure was not contract-invalidating):
    # plan the best current package and carry on with the turn as decided.
    chosen = packages[0]
    state.selected_package_id = chosen.package_id
    set_ids = list(chosen.set_ids or ([chosen.set_id] if chosen.set_id else []))
    state.selected_package_set_id = set_ids[0] if set_ids else None
    state.selected_package_set_ids = set_ids
    state.selected_package_label = chosen.label
    state.selected_package_price_cents = chosen.price_cents
    state.confirmed_budget_cents = chosen.price_cents
    state.status = FanStatus.OFFER_SELECTED
    await save_fan_state(fan_id, creator_id, state)

    replan = await plan_session_for_fan(
        creator_id,
        fan_id,
        selected_set_ids=set_ids,
        selected_price_cents=chosen.price_cents,
    )
    if replan.get("status") == "ok":
        return PlanRecovery(
            status=status,
            failure_class=failure_class,
            session=replan.get("session"),
            replacement_packages=packages,
            reason="replanned_from_current_inventory",
        )

    # One attempt, then stop. A recovery loop that keeps re-planning against the
    # same inventory would burn the turn instead of answering the fan.
    state.status = FanStatus.OFFER_PENDING
    state.selected_package_id = None
    state.selected_package_set_id = None
    state.selected_package_set_ids = []
    state.selected_package_price_cents = None
    state.confirmed_budget_cents = None
    await save_fan_state(fan_id, creator_id, state)
    return PlanRecovery(
        status=str(replan.get("status") or status),
        failure_class=failure_class,
        replacement_packages=packages,
        present_replacement=True,
        accepted_contract_lost=had_accepted_contract,
        reason="replan_failed_presenting_options",
    )
