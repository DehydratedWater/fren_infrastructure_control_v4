"""CLI entrypoint for the website_monitor tool."""

from app.tools.research.website_monitor import WebsiteMonitorTool

if __name__ == "__main__":
    WebsiteMonitorTool.run()
