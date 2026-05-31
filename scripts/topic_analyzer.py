"""CLI entrypoint for the topic_analyzer tool."""

from app.tools.research.topic_analyzer import TopicAnalyzerTool

if __name__ == "__main__":
    TopicAnalyzerTool.run()
