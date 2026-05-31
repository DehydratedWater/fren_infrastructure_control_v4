#!/usr/bin/env python3
"""CLI wrapper for QuestionSenderTool."""

from app.tools.telegram.question_sender import QuestionSenderTool

if __name__ == "__main__":
    QuestionSenderTool.run()
