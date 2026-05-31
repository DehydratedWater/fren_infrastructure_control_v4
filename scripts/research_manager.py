"""CLI entrypoint for the research_manager tool."""

from app.tools.research.research_manager import ResearchManagerTool

if __name__ == "__main__":
    ResearchManagerTool.run()
