"""Every API Fansly call site must be accounted for.

Credit observability is only useful if it is complete. The failure this guards
is not a wrong number — it is a call site that quietly bypasses ``request()``
and therefore appears nowhere in the usage report, so spend looks lower than it
is exactly when somebody is trying to find out where it went.

These are source-level assertions on purpose. A behavioural test can only see
the paths a test happens to exercise; the whole point here is to catch the path
nobody thought about.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Files that make raw httpx calls against the provider.
SOURCES = [ROOT / "main.py", ROOT / "services" / "suggestions.py"]

# A raw provider call looks like `client.<verb>(\n  apifansly_url(...)`.
RAW_CALL = re.compile(r"\b(?:await\s+)?[\w().]*\.(?:post|get|put|delete)\(\s*$")

ACCOUNTING_CALLS = (
    "record_apifansly_raw_call",
    "raise_for_apifansly_response",
)

# How far after the call the accounting may appear. Generous enough for the
# argument list plus a comment, tight enough that it has to belong to the call.
ACCOUNTING_WINDOW_LINES = 40


def _raw_call_sites() -> list[tuple[Path, int, str]]:
    sites: list[tuple[Path, int, str]] = []
    for path in SOURCES:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "apifansly_url(" not in line:
                continue
            # The verb is on the previous line in this codebase's formatting.
            previous = lines[index - 1] if index else ""
            if not RAW_CALL.search(previous.rstrip()):
                continue
            sites.append((path, index + 1, "\n".join(lines[index : index + ACCOUNTING_WINDOW_LINES])))
    return sites


def test_the_scan_actually_finds_the_known_raw_call_sites():
    """Guard the guard: a regex that matches nothing would pass silently."""
    sites = _raw_call_sites()
    assert len(sites) >= 5, [(str(path), line) for path, line, _ in sites]


@pytest.mark.parametrize(
    "path,line,window",
    [
        pytest.param(path, line, window, id=f"{path.name}:{line}")
        for path, line, window in _raw_call_sites()
    ],
)
def test_every_raw_provider_call_is_accounted_for(path, line, window):
    assert any(call in window for call in ACCOUNTING_CALLS), (
        f"{path.name}:{line} calls API Fansly without recording usage. "
        "Add record_apifansly_raw_call (or raise_for_apifansly_response) so "
        "the credit it costs appears in /apifansly-usage."
    )


def test_nothing_asks_the_chat_endpoint_for_more_than_ten_messages():
    """API Fansly documents `limit min=1 max=10`. Asking for 50 returns 10.

    A request for 50 is not harmless optimism: it makes an import believe it is
    buying five times the messages per page that it is, which is how the old
    /load-history quietly paid for 500 pages while reporting 100.

    Scoped to `list_chat_messages` deliberately — other endpoints (vault media,
    chat groups on the direct Fansly client) have their own, larger limits.
    """
    offenders: list[str] = []
    for path in list(ROOT.glob("*.py")) + list((ROOT / "services").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"list_chat_messages\s*\(", source):
            call = source[match.end() : match.end() + 400]
            limit = re.search(r"limit\s*=\s*(\d+)", call)
            if limit and int(limit.group(1)) > 10:
                offenders.append(f"{path.name}: limit={limit.group(1)}")
    assert offenders == [], (
        "these call the chat-messages endpoint above its documented maximum: "
        + "; ".join(offenders)
    )


def test_the_page_ceiling_is_stated_once_and_documented():
    from services.apifansly import CHAT_MESSAGE_PAGE_MAX

    assert CHAT_MESSAGE_PAGE_MAX == 10
    source = (ROOT / "services" / "apifansly.py").read_text(encoding="utf-8")
    assert "DO NOT RAISE THIS" in source
    assert "min=1 max=10" in source


def test_the_webhook_route_counts_received_events():
    """80 events is a credit, so an uncounted webhook is an invisible cost."""
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    webhook = source[source.index("async def fansly_webhook") :][:3_000]
    assert "record_apifansly_webhook_event" in webhook


def test_history_backfill_contains_no_media_download_path():
    """The rule that keeps an archive import off the 2 credits/MB meter."""
    for name in ("fan_history.py", "fan_history_memory.py"):
        source = (ROOT / "services" / name).read_text(encoding="utf-8")
        assert "download_media" not in source, name
        assert "media/download" not in source, name


def test_history_backfill_sends_nothing_to_the_platform():
    """Historical import writes locally. It never reaches the fan's inbox.

    Matched against imported names and URL paths rather than free text, so a
    docstring explaining the rule does not trip the rule.
    """
    forbidden = (
        "send_apifansly_message",
        "send_message(",
        "/typing",
        "mark-as-read",
        "delete_message",
    )
    for name in ("fan_history.py", "fan_history_memory.py"):
        source = (ROOT / "services" / name).read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"{name} references {marker}"
