"""CLI entrypoint for the calendar_manager tool."""

from app.tools.comms.calendar_manager import CalendarManagerTool

if __name__ == "__main__":
    CalendarManagerTool.run()
