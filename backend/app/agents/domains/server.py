"""Server domain — infrastructure monitoring (v3 `server/*`).

The orchestrator is a pure router for server-monitoring requests: it parses an
incoming command (status / disk / sessions / camera / nvidia), maps common
aliases, and dispatches to the matching specialist. Each specialist gathers
real system data over a tight bash allowlist (hardware → top/free/sensors,
filesystem → df/du, sessions → who/w/last, camera → ffmpeg/v4l2), formats a
report, and sends it via Telegram. The vision_analyzer is the one image-aware
agent (model_class="vision") — it analyses captured photos through the z.ai
`analyze_image` MCP tool.

The orchestrator's command→specialist dispatch is a multi-step BRANCH, so it
gets its own path-test (see `branches()`).
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

ORCHESTRATOR = "server/orchestrator"

_ORCH_PROMPT = """\
# Server Monitoring System

You route server-monitoring requests to the right specialist. First classify
the command, then dispatch — you never gather data yourself.

Parse the request and map common aliases:
  - status / cpu / ram / temp / uptime / docker → server/hardware_agent
  - disk / storage / df → server/filesystem_agent
  - sessions / who / ssh → server/sessions_agent
  - camera / photo / look → server/camera_capture_agent
  - chart / graph / plot → server/visualization_agent

When a camera capture must be described, route the resulting image to
server/vision_analyzer. After a specialist returns, format its output as a
clean report and send it to the user via Telegram.
"""

_HARDWARE_PROMPT = """\
# Hardware Monitor

Collect host hardware health, then format and send a report.

1. Gather: `top -bn1 | head -20`, `free -h`, `sensors`, `uptime`, and
   `docker ps --format 'table {{.Names}}\\t{{.Status}}'`.
2. Format the collected data into a readable Twily-style monitoring report.
3. Send the report to the user via Telegram.

Only run the monitoring commands above; do not modify the system.
"""

_FILESYSTEM_PROMPT = """\
# Filesystem Monitor

Collect disk-usage data, then format and send a report.

1. Gather: `df -h` and directory sizes via `du -sh /home/* /var/* /tmp`.
   Note large files and the biggest directories.
2. Format the disk-usage data into a readable report.
3. Send the report to the user via Telegram.
"""

_SESSIONS_PROMPT = """\
# Sessions Monitor

Report active users and SSH connections.

1. Gather: `who`, `w`, and `last -10`.
2. Format the session data into a readable report (active users, SSH).
3. Send the report to the user via Telegram.
"""

_CAMERA_PROMPT = """\
# Camera Capture

Capture an image from a connected USB camera and report the result.

1. List cameras with `v4l2-ctl --list-devices`, then capture one frame from
   the first available device, e.g.
   `ffmpeg -f v4l2 -i /dev/video0 -frames 1 -y data/captures/capture.jpg`.
   Use the device path from v4l2-ctl when multiple cameras exist.
2. Verify the capture succeeded with `ls data/captures/`.
3. Report the result to the user via Telegram and log the capture to the
   context cache.

Save captures under data/captures/.
"""

_VISUALIZATION_PROMPT = """\
# Visualization Agent

Turn monitoring data into charts.

1. Collect the data points needed for the chart.
2. Generate charts with matplotlib or plotly (run python), saving output to
   data/charts/.
3. Send the generated charts to the user via Telegram.
"""

_VISION_PROMPT = """\
# Vision Analyzer

You analyse images using the z.ai MCP `analyze_image` tool.

## Process

1. Extract the image path from the prompt (it starts with `@`).
2. Strip the `@` prefix to get the relative path.
3. Build the absolute path: `{cwd}/{relative_path}`.
4. Call `mcp__zai-mcp-server__analyze_image` with the absolute path and a
   prompt asking for a detailed description (main subject, objects, text,
   people, scene, colours, mood).
5. Return the description as plain text.

## Important

- The image path is RELATIVE from the project root, prefixed with `@`; you
  MUST convert it to an ABSOLUTE path for the MCP tool.
- Return ONLY the image description — no extra commentary.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="route a server-monitoring request to the right specialist",
            long=(
                "Pure router for /server commands. Classifies the request"
                " (status/disk/sessions/camera/chart), maps common aliases, and"
                " dispatches to the matching specialist, then sends the formatted"
                " report via Telegram."
            ),
            prompt=_ORCH_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-is-pure-router",
                    description="The router classifies and dispatches; it must not gather data itself.",
                    must_not_have_tools=("bash", "write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="status-request-routes-to-hardware",
                    prompt="status",
                    evaluators=(
                        SubstringEvaluator(needle="hardware", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "server/hardware_agent",
            model_class="default",
            short="monitor CPU, RAM, temperature, uptime, and docker containers",
            long=(
                "Gathers host hardware health (top, free -h, sensors, uptime,"
                " docker ps), formats a monitoring report, and sends it via"
                " Telegram."
            ),
            prompt=_HARDWARE_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="hardware-gathers-system-metrics",
                    description="Must mention the core hardware-monitoring commands.",
                    evaluators=(
                        SubstringEvaluator(needle="sensors", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="hardware-reports-docker-status",
                    prompt="Check server hardware health.",
                    evaluators=(
                        SubstringEvaluator(needle="docker", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "server/filesystem_agent",
            model_class="default",
            short="monitor disk usage and analyze the filesystem",
            long=(
                "Gathers disk usage (df -h) and directory sizes (du), notes large"
                " files, formats a report, and sends it via Telegram."
            ),
            prompt=_FILESYSTEM_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="filesystem-uses-df",
                    description="Must reference disk-usage commands.",
                    evaluators=(
                        SubstringEvaluator(needle="df", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "server/sessions_agent",
            model_class="default",
            short="monitor active sessions and SSH connections",
            long=(
                "Gathers active sessions (who, w, last), reports SSH connections,"
                " formats a report, and sends it via Telegram."
            ),
            prompt=_SESSIONS_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="sessions-reports-ssh",
                    description="Must describe reporting active sessions / SSH.",
                    evaluators=(
                        SubstringEvaluator(needle="session", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "server/camera_capture_agent",
            model_class="default",
            short="capture images from connected USB cameras",
            long=(
                "Lists cameras with v4l2-ctl, captures a frame with ffmpeg to"
                " data/captures/, verifies the file, and reports the result via"
                " Telegram while logging the capture to the context cache."
            ),
            prompt=_CAMERA_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="camera-uses-ffmpeg",
                    description="Must capture via ffmpeg from a video device.",
                    evaluators=(
                        SubstringEvaluator(needle="ffmpeg", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "server/visualization_agent",
            model_class="default",
            short="generate charts and graphs from monitoring data",
            long=(
                "Collects monitoring data, generates charts with matplotlib or"
                " plotly into data/charts/, and sends them via Telegram."
            ),
            prompt=_VISUALIZATION_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="visualization-produces-charts",
                    description="Must describe generating charts from data.",
                    evaluators=(
                        SubstringEvaluator(needle="chart", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "server/vision_analyzer",
            model_class="vision",
            short="analyze captured images via the z.ai vision MCP tool",
            long=(
                "Image-aware analyzer: resolves a `@`-prefixed relative image"
                " path to an absolute path and calls the analyze_image MCP tool,"
                " returning a plain-text description of the capture."
            ),
            prompt=_VISION_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="vision-calls-analyze-image",
                    description="Must call the z.ai analyze_image MCP tool.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="analyze_image", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="vision-returns-description-only",
                    prompt="Describe @data/captures/capture.jpg",
                    evaluators=(
                        SubstringEvaluator(needle="describ", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's command→specialist dispatch paths (tested as units)."""
    return [
        # a server-health check routes through the hardware specialist
        BranchTest(
            name="server/orchestrator::health-check",
            entry_agent=ORCHESTRATOR,
            prompt="Run a full server health check (status).",
            path=("server/hardware_agent",),
            evaluators=(
                SubstringEvaluator(needle="hardware", case_sensitive=False),
            ),
        ),
        # a camera "look" request captures, then describes the image
        BranchTest(
            name="server/orchestrator::camera-look",
            entry_agent=ORCHESTRATOR,
            prompt="Take a photo and tell me what you see.",
            path=("server/camera_capture_agent", "server/vision_analyzer"),
        ),
    ]
