"""Twily's world — an autonomous roleplay life-sim she plays in the background.

A self-contained subsystem (its own FastAPI app on a separate port, its own
scheduler-driven turn loop) that gives the Twily persona a *life*: rooms she
lives/cooks/works in, a town she roams, NPCs who shape her, and a computer she
researches the real web on (via the existing SearchAPI). Her experiences feed
back into her persona memory, so who she is on Telegram is shaped by who she is
here.

Design is inspired by /home/dw/programing/roll_play_learning (shapes: world
package format, the pre-classify → generate → extract-world-update → persist
turn pipeline, the vanilla-JS observe/play UI) but is built fresh and
fren-native: Postgres (not SQLite), the local qwen via src.interactive, the
existing SearchAPI tool, and the persona-memory repos. The world itself is an
invented modern-Ponyville blend (`packages/twily_haven`), not an rpl package.
"""

from __future__ import annotations
