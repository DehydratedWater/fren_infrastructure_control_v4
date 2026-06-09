"""Delivery-quality gate — autoloop-optimised policy for outbound messages.

Every Twily send (reactive + proactive) flows through
`app.tools.telegram.send_message`, which consults this package before
anything reaches Telegram. The gate is a PURE function over a JSON-able
policy dict so the framework's autoresearch loop can probe and tune it
(`app.delivery.gate_probes.improve_gate`), promoting winners under
component_id "policy:delivery_gate" in `.oac/promoted/`.
"""

from app.delivery.gate import (
    DEFAULT_POLICY,
    GateDecision,
    active_policy,
    evaluate_message,
)

__all__ = [
    "DEFAULT_POLICY",
    "GateDecision",
    "active_policy",
    "evaluate_message",
]
