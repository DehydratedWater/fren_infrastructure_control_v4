#!/usr/bin/env python3
"""Camera capture — grab a frame from webcam or desk camera.

Usage:
    uv run scripts/camera_capture.py --command webcam
    uv run scripts/camera_capture.py --command desk
    uv run scripts/camera_capture.py --command both
"""

from app.tools.system.camera_capture import CameraCaptureTool

if __name__ == "__main__":
    CameraCaptureTool.run()
