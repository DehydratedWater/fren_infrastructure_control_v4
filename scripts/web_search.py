"""CLI entrypoint for the web_search tool."""

from app.tools.research.web_search import WebSearchTool

if __name__ == "__main__":
    WebSearchTool.run()
