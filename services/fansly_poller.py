"""
services/fansly_poller.py

Runs one asyncio task per model account that polls for new messages.
When a fan sends a message, it triggers your existing Cleopatra AI pipeline.
"""

import asyncio
import logging
import random
import time
from typing import Callable, Optional

from .fansly_session_store import SessionStore
from .fansly_client import SessionExpiredError

logger = logging.getLogger(__name__)


class FanslyPoller:
    """
    Background polling manager for all model accounts.

    Wire into your FastAPI lifespan:

        poller = FanslyPoller(
            session_store=session_store,
            on_new_message=handle_new_message,
        )
        await poller.start_all()   # on startup
        await poller.stop_all()    # on shutdown

    Your handler receives inbound fan messages and should feed them
    into your existing AI suggestion pipeline.
    """

    def __init__(
        self,
        session_store: SessionStore,
        on_new_message: Callable,   # async fn(account_id, group_id, message_dict)
    ):
        self._store = session_store
        self._on_new_message = on_new_message
        # account_id → asyncio.Task
        self._tasks: dict[str, asyncio.Task] = {}
        # account_id → {group_id → last_message_id}
        self._cursors: dict[str, dict[str, str]] = {}

    async def start_all(self):
        """Start polling for every session currently in the store."""
        for account_id in self._store._cache:
            await self.start(account_id)
        logger.info(f"Polling started for {len(self._tasks)} accounts")

    async def start(self, account_id: str):
        if account_id in self._tasks:
            return
        task = asyncio.create_task(
            self._poll_loop(account_id),
            name=f"fansly-poll-{account_id}"
        )
        self._tasks[account_id] = task
        logger.info(f"[{account_id}] Poll task started")

    async def stop(self, account_id: str):
        task = self._tasks.pop(account_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def stop_all(self):
        for account_id in list(self._tasks):
            await self.stop(account_id)

    async def _poll_loop(self, account_id: str):
        # Step 1: seed cursors (don't fire callbacks on existing messages)
        await self._seed_cursors(account_id)

        last_activity = time.time()

        while True:
            try:
                had_new = await self._poll_once(account_id)
                self._store.record_success(account_id)
                if had_new:
                    last_activity = time.time()

            except SessionExpiredError as e:
                await self._store.record_failure(account_id, str(e))
                # Stop polling this account — it needs manual re-auth
                logger.error(f"[{account_id}] Session dead, stopping poll")
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                await self._store.record_failure(account_id, str(e))
                await asyncio.sleep(30)
                continue

            await asyncio.sleep(self._interval(last_activity))

    async def _seed_cursors(self, account_id: str):
        """On startup: record latest message ID per group without firing callbacks."""
        try:
            async with self._store.get_client(account_id) as client:
                groups = await client.get_chat_groups(limit=30)
                self._cursors[account_id] = {}
                for g in groups:
                    gid = g["id"]
                    data = await client.get_messages(gid, limit=1)
                    msgs = data.get("messages", [])
                    if msgs:
                        self._cursors[account_id][gid] = msgs[0]["id"]
                logger.info(f"[{account_id}] Seeded {len(groups)} group cursors")
        except Exception as e:
            logger.error(f"[{account_id}] Cursor seed failed: {e}")
            self._cursors[account_id] = {}

    async def _poll_once(self, account_id: str) -> bool:
        """Poll all groups for new messages. Returns True if any new messages found."""
        had_new = False
        cursors = self._cursors.setdefault(account_id, {})

        async with self._store.get_client(account_id) as client:
            groups = await client.get_chat_groups(limit=30)

            for g in groups:
                gid = g["id"]
                last_id = cursors.get(gid)

                if not last_id:
                    # New group — seed it
                    data = await client.get_messages(gid, limit=1)
                    msgs = data.get("messages", [])
                    if msgs:
                        cursors[gid] = msgs[0]["id"]
                    continue

                new_msgs = await client.poll_new_messages(gid, last_id)
                if not new_msgs:
                    continue

                # Update cursor to newest
                cursors[gid] = new_msgs[0]["id"]
                had_new = True

                # Fire callback for each inbound fan message
                # Skip messages sent BY the model (senderId == account_id)
                for msg in reversed(new_msgs):  # oldest first
                    if msg.get("senderId") == account_id:
                        continue
                    try:
                        await self._on_new_message(account_id, gid, msg)
                    except Exception as e:
                        logger.error(f"[{account_id}] Handler error: {e}")

        return had_new

    @staticmethod
    def _interval(last_activity: float) -> float:
        """
        Adaptive poll interval based on how long the account has been quiet.
        Adds ±20% jitter so multiple accounts don't poll simultaneously.
        """
        silence = time.time() - last_activity
        if silence < 300:        # active in last 5 min
            base = random.uniform(8, 15)
        elif silence < 1800:     # idle 5–30 min
            base = random.uniform(30, 60)
        else:                    # quiet 30+ min
            base = random.uniform(120, 300)
        return base * random.uniform(0.8, 1.2)