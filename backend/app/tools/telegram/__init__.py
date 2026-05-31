"""Telegram script-tools: message/image/video/file/voice delivery, questions, guidance.

Ported from v3's ``fren.tools.telegram``. Each module is a ``ScriptTool`` with a
matching CLI entrypoint under ``scripts/<name>.py``:

- ``send_message``   — text delivery (style scorer, dedup, paragraph split, TTS spawn)
- ``send_image``     — photo delivery + chat-history @path save
- ``send_video``     — video delivery (50 MB limit) + chat-history @path save
- ``send_file``      — document delivery
- ``send_voice``     — TTS synthesis + Telegram voice message
- ``question_sender``— inline-keyboard questions with rate limiting + dedup
- ``emit_guidance``  — PersonaGuidance delivery channel (ack fast-path + persona_prose)
"""
