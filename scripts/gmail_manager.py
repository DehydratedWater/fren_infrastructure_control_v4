"""CLI entrypoint for the gmail_manager tool."""

from app.tools.comms.gmail_manager import GmailManagerTool

if __name__ == "__main__":
    GmailManagerTool.run()
