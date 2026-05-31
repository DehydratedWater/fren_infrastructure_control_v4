"""Telegram shared helpers used by tools and (later) the bot runtime.

Ported from v3's ``fren.telegram`` package. Only the helper modules that
tools or other code import are present so far — bot.py / handlers.py and the
bot-runtime files (scheduler, vibe_chart, rp_bot/rp_handlers/rp_charts) are a
later step. Current modules:

- ``state``        — bot state persistence (mode/model/content-mode, model tags)
- ``rp_prose``     — RP prose-generation helpers (provider config, env expansion)
- ``persona_prose``— Twily-voice rendering + chat-context fetch + guidance parsing
"""
