-- Payment lifecycle enum values introduced by the OFFER_SELECTED /
-- PAYMENT_PENDING state contract. Safe to apply repeatedly.

alter type public.fan_commercial_status
    add value if not exists 'OFFER_SELECTED';

alter type public.fan_commercial_status
    add value if not exists 'PAYMENT_PENDING';
