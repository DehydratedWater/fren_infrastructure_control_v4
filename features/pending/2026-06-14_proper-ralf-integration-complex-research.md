# Proper Ralf Integration for Complex Multi-Step Research

**Date:** 2026-06-14
**Priority:** high

## Description
The user wants to be able to use Ralf for complex research tasks directly instead of running investigator by itself. Currently, when the user says "use Ralf", the system sometimes just runs investigator without the full Ralf planning pipeline. The user requests that Ralf should:
1. Break the task into stages
2. Plan them
3. Execute them
4. Send results by email at the end

The desired behavior is to have proper Ralf integration as the default for complex multi-step research tasks.

## Motivation
The user is experiencing inconsistent behavior when requesting Ralf - sometimes the system uses Ralf properly (multi-stage pipeline), but other times it just runs investigator directly. This defeats the purpose of Ralf's sophisticated multi-stage research capabilities. The user wants reliable, predictable behavior where complex research tasks automatically use Ralf's full planning and execution pipeline, with final results delivered via email.

## Affected Components

### Scripts
- `/scripts/ralf_spawn.py` - Ralf spawning script (9056 bytes) - handles Ralf session initialization and stage management
- `/scripts/ralf_manager.py` - Ralf manager script - likely handles Ralf orchestration
- `/scripts/ralf_cleanup.py` - Ralf cleanup script (1122 bytes) - handles Ralf session cleanup
- `/scripts/research_manager.py` - Research manager script - currently used for basic investigation

### Backend Agent Domains
- `/backend/app/agents/domains/research.py` (29547 bytes) - Research domain handling
- `/backend/app/agents/domains/investigation.py` (3852 bytes) - Investigation domain handling  
- `/backend/app/agents/domains/workflow_master.py` (15319 bytes) - Workflow orchestration
- `/backend/app/agents/workflows.py` (84008 bytes) - Workflow definitions and routing

### Backend Tools - Research
- `/backend/app/tools/research/research_manager.py` (9629 bytes) - Research manager implementation
- `/backend/app/tools/research/topic_analyzer.py` (9429 bytes) - Topic analysis for research
- `/backend/app/tools/research/web_search.py` (3279 bytes) - Web search functionality
- `/backend/app/tools/research/document_manager.py` (11434 bytes) - Document management

### Backend Tools - Communications
- `/backend/app/tools/comms/gmail_manager.py` (6662 bytes) - Email sending functionality
- `/backend/app/tools/comms/calendar_manager.py` (6768 bytes) - Calendar management

### Backend Agents Core
- `/backend/app/agents/config.py` (7000 bytes) - Agent configuration and routing
- `/backend/app/agents/_tools.py` (24918 bytes) - Tool definitions and integration

### Session Evidence
From session inspection, multiple Ralf sessions show the multi-stage pattern:
- `ses_1477e891cffe7YgHRzkGXOpAUr` - "Ralf ID stage 4 attempt 1"
- `ses_1477e891cffe7YgHRzkGXOpAUr` - "Ralf stage 4 attempt 2"  
- `ses_147ae70e0ffefN3OleTwrpUXiF` - "ralf_id stage 2"
- `ses_147afbf46ffe3NA7hgQIxwUbqr` - "ralf_20260611_182210_1f515b29"

These sessions demonstrate Ralf's intended multi-stage architecture (stage 1, 2, 4, etc.) which is not being consistently triggered.

## Suggested Implementation

### Phase 1: Improve Routing Logic
**Location:** `/backend/app/agents/config.py` and `/backend/app/agents/domains/research.py`

1. Add intent detection for "complex multi-step research" vs "simple investigation"
2. Modify routing logic to:
   - Automatically trigger Ralf spawn when complex research is detected
   - Fall back to investigator only for simple, single-query research
   - Consider keywords like "use Ralf", "comprehensive", "multi-stage", "deep dive" as explicit Ralf requests

### Phase 2: Enhance Ralf Pipeline  
**Location:** `/scripts/ralf_spawn.py` and `/scripts/ralf_manager.py`

1. Add explicit stage planning before execution:
   - Stage 1: Topic analysis and research question formulation
   - Stage 2: Multi-source information gathering (web, documents, specialized sources)
   - Stage 3: Synthesis and analysis
   - Stage 4: Final report generation
   - Stage 5: Email delivery

2. Add email delivery integration:
   - Use existing `/backend/app/tools/comms/gmail_manager.py`
   - Format final research output as structured email
   - Include session ID and timestamp for tracking

### Phase 3: Integration Points
**Location:** `/backend/app/agents/domains/workflow_master.py`

1. Add Ralf as a workflow type in the workflow master
2. Ensure Ralf workflow includes:
   - Stage-by-stage progress updates
   - Email delivery as final step
   - Error handling and stage retry logic
   - User notification at each stage

### Phase 4: Tool Integration
**Location:** `/backend/app/agents/_tools.py`

1. Add explicit Ralf tool definition
2. Connect ralf_manager.py as a callable tool
3. Ensure tool parameters match expected Ralf workflow inputs

## Acceptance Criteria

1. **Routing Accuracy**: When user says "use Ralf" or requests complex research, Ralf is consistently triggered 100% of the time
2. **Stage Breakdown**: Research tasks are automatically broken into 4-5 logical stages before execution
3. **Email Delivery**: Research results are automatically sent via email using gmail_manager.py
4. **Progress Visibility**: Each stage completion is reported to the user
5. **Backward Compatibility**: Simple research queries can still use investigator directly
6. **Error Handling**: Failed stages trigger appropriate retry logic and user notification
7. **Session Tracking**: Ralf sessions include proper metadata (stage number, research topic, completion status)