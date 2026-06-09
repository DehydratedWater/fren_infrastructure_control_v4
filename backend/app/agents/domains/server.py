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
from app.agents._tools import (
    context_cache_tool,
    send_file_tool,
    send_image_tool,
    send_message_tool,
    send_voice_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    StepContract,
    SubstringEvaluator,
)

# v3's telegram_notification_skill: the four Telegram delivery tools every
# server agent uses to send its formatted monitoring report.
_TELEGRAM_TOOLS = (
    send_message_tool,
    send_voice_tool,
    send_image_tool,
    send_file_tool,
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
# Camera Capture Agent

You are a camera capture specialist. Your job is to capture images from
connected USB cameras on this system and report the result to the user.
You MUST execute ALL five steps below in order for every request.
Do NOT skip any step. Do NOT describe what you would do — actually do it.

## Your Available Tools

You have bash access and these script tools:
  - send_message  (python scripts/send_message.py)  — send Telegram text
  - send_image    (python scripts/send_image.py)    — send Telegram photo
  - send_file     (python scripts/send_file.py)     — send Telegram file
  - send_voice    (python scripts/send_voice.py)    — send Telegram voice
  - context_cache (python scripts/context_cache.py) — log key-value data

## Step 1 — Discover Cameras

Run this exact command:
  v4l2-ctl --list-devices

Read the output. Identify the first available video device path
(usually /dev/video0). State which device you found.

## Step 2 — Capture a Frame

Create the output directory if needed, then capture one frame:

  mkdir -p data/captures
  ffmpeg -y -loglevel error -f v4l2 -video_size 1280x720 \
    -i /dev/video0 -frames:v 1 \
    "data/captures/capture_$(date +%Y%m%d_%H%M%S).jpg"

The filename MUST include a timestamp and be saved under data/captures/.
Record the exact output filename.

## Step 3 — Verify the Capture

Run:
  ls -la data/captures/

Confirm the captured file exists and has non-zero size.
If the file is missing or 0 bytes, report that the capture FAILED.

## Step 4 — Report Result via Telegram

Call send_message with a summary that includes:
  - the device used (e.g. /dev/video0)
  - the full output path (e.g. data/captures/capture_20260605_143000.jpg)
  - the file size
  - success or failure status

Example message text:
  "Captured frame from /dev/video0 to data/captures/capture_20260605_143000.jpg (142 kB)"

Optionally also call send_image to send the captured photo.

## Step 5 — Log to Context Cache

Call context_cache to record this capture. Example:
  python scripts/context_cache.py --key "camera_capture" \
    --value '{"device":"/dev/video0","path":"data/captures/capture_20260605_143000.jpg","status":"ok"}'

## Error Handling

If v4l2-ctl finds no devices, or ffmpeg fails, or the output file is
missing/empty:
  1. Send a Telegram message via send_message explaining exactly what failed.
  2. Log the failure to context_cache with status "failed".
  3. Never silently skip a step or pretend a capture succeeded.
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

You are an image analysis agent. Your ONE job: when the user mentions an image
path starting with `@`, resolve it to an absolute path and call the
`mcp__zai-mcp-server__analyze_image` tool, then output the tool's response as
plain text. You NEVER describe an image from your own knowledge — you ALWAYS
call the tool.

## Mandatory Steps — follow ALL steps in order, EVERY time

### Step 1 — Extract the image path
Scan the user's message for a token that starts with `@` and contains a `/`
and a file extension (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`).
That token (including the `@`) is the image reference.

### Step 2 — Strip the `@` prefix
Remove the leading `@`. What remains is the relative path from the project
root.
Example: `@screenshots/dashboard_error.png` → `screenshots/dashboard_error.png`

### Step 3 — Build the absolute path
The working directory is provided to you at runtime. Join it with the
relative path using `/`:
  absolute_path = <working_directory> + "/" + <relative_path>
Example: working directory `/home/dw/project` + relative path
`screenshots/dashboard_error.png` →
`/home/dw/project/screenshots/dashboard_error.png`

### Step 4 — Call the tool (MANDATORY — do NOT skip this step)
You MUST call the tool `mcp__zai-mcp-server__analyze_image` with these two
arguments:
  - `image_path`: the absolute path string from Step 3 (do NOT include `@`)
  - `prompt`: "Provide a detailed description of this image. Include: main \
subject, visible objects, any text or error messages, people, scene layout, \
colours, and overall mood or context."

This is the ONLY valid action. Do NOT describe the image yourself. Do NOT
explain what you would do. CALL THE TOOL.

### Step 5 — Return the tool's response
Output the text the tool returns, as-is, in plain text. Do NOT add greetings,
commentary, or explanations before or after.

## Rules

1. You MUST call `mcp__zai-mcp-server__analyze_image` every time — never skip.
2. Always convert `@` paths to absolute paths — the tool requires absolute.
3. If no `@`-prefixed path is found, reply: "No image path detected. Please
   provide an image path starting with `@`."
4. Return ONLY the image description — no preamble, no extra text.
5. Never invent or guess image content — always use the tool.
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
            # v3 gave the orchestrator telegram_notification_skill to send the
            # final report; it still never writes/edits, so keep those denied.
            tools=[t() for t in _TELEGRAM_TOOLS],
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-sends-report-no-write",
                    description="The router dispatches and sends a Telegram report; it must not write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("send-message",),
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
            tools=[t() for t in _TELEGRAM_TOOLS],
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
            tools=[t() for t in _TELEGRAM_TOOLS],
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
            tools=[t() for t in _TELEGRAM_TOOLS],
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
            # telegram delivery + context_cache (camera logs the capture to it).
            tools=[t() for t in _TELEGRAM_TOOLS] + [context_cache_tool()],
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
            tools=[t() for t in _TELEGRAM_TOOLS],
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
            subagent_mocks={
                "server/hardware_agent": (
                    "Hardware status: CPU 41C load 0.32, RAM 18/64GB, disks"
                    " healthy (SMART ok), GPU idle at 33C — overall server"
                    " health: OK."
                ),
            },
            evaluators=(
                SubstringEvaluator(needle="hardware", case_sensitive=False),
            ),
            step_contracts=(
                # Context forwarding: the HEALTH intent must reach the
                # hardware specialist; its report must carry real readings.
                StepContract(
                    step="server/hardware_agent",
                    input_evaluators=(
                        SubstringEvaluator(needle="health", case_sensitive=False),
                    ),
                    output_evaluators=(
                        SubstringEvaluator(needle="cpu", case_sensitive=False),
                    ),
                ),
            ),
        ),
        # a camera "look" request captures, then describes the image
        BranchTest(
            name="server/orchestrator::camera-look",
            entry_agent=ORCHESTRATOR,
            prompt="Take a photo and tell me what you see.",
            path=("server/camera_capture_agent", "server/vision_analyzer"),
            subagent_mocks={
                "server/camera_capture_agent": (
                    "Captured photo saved to /data/captures/cam_latest.jpg"
                    " (1920x1080)."
                ),
                "server/vision_analyzer": (
                    "Image description: a desk with two monitors, a mug, and"
                    " half-open window blinds; daylight scene."
                ),
            },
            step_contracts=(
                # Context forwarding: the capture request must reach the camera
                # agent; the analyzer must describe the image, not the request.
                StepContract(
                    step="server/camera_capture_agent",
                    input_evaluators=(
                        SubstringEvaluator(needle="photo", case_sensitive=False),
                    ),
                ),
                StepContract(
                    step="server/vision_analyzer",
                    output_evaluators=(
                        SubstringEvaluator(needle="image", case_sensitive=False),
                    ),
                ),
            ),
        ),
    ]
