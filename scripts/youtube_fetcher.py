"""CLI entrypoint for the youtube_fetcher tool."""

from app.tools.research.youtube_fetcher import YouTubeFetcherTool

if __name__ == "__main__":
    YouTubeFetcherTool.run()
