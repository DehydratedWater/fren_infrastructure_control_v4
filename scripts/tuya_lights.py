"""CLI entrypoint for the tuya_lights tool."""

from app.tools.home.tuya_lights import TuyaLightsTool

if __name__ == "__main__":
    TuyaLightsTool.run()
