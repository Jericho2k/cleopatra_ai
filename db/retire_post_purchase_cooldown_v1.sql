-- Retire the fixed post-purchase message cooldown.
--
-- `creator_commercial_policies.post_purchase_cooldown_messages` configured a
-- "wait N creator messages after a purchase before offering again" window that
-- the move to one-unlock sessions made unreachable: a single-step plan reaches
-- `completed` on the purchase, and the completed branch of
-- services/session_lifecycle.mark_step_purchased cleared the counter on the
-- same line that would have set it. The dial was live in the dashboard, stored
-- per creator, and did nothing.
--
-- What replaces it is services/experience_director.py: a scene that survives
-- the commercial session and re-opens offer discovery when the dialogue has
-- actually produced a bridge — immediately if he asks for more, and never on a
-- flat reaction however long he keeps talking.
--
-- DROP rather than leave in place, because an operator-visible control that
-- changes nothing is worse than no control. Idempotent; the column carries
-- configuration, not history, so nothing is lost.

alter table public.creator_commercial_policies
    drop column if exists post_purchase_cooldown_messages;
