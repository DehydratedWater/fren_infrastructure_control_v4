"""CLI entrypoint for the document_manager tool."""

from app.tools.research.document_manager import DocumentManagerTool

if __name__ == "__main__":
    DocumentManagerTool.run()
