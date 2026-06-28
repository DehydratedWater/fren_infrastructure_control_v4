# Feature: Proper Ralf Integration for Complex Multi-Step Research

**Date:** 2025-06-14  
**Priority:** High

## Description

Currently, when users invoke Ralf for complex research tasks, the system sometimes runs investigator directly without using the full Ralf planning pipeline. This results in suboptimal research execution that lacks the structured multi-stage approach that Ralf is designed to provide.

The system should properly integrate Ralf as the default handler for complex multi-step research tasks, ensuring that:

1. Tasks are broken down into logical stages
2. Each stage is planned and executed in sequence
3. Results are compiled and sent via email at completion
4. The full Ralf planning pipeline is utilized rather than bypassing to investigator

## Motivation

Ralf is designed to handle complex, multi-step research tasks through a structured planning and execution pipeline. However, the current integration is inconsistent - sometimes the full pipeline is used, other times investigator is invoked directly without the planning stages.

This inconsistency leads to:
- Poor research quality for complex tasks that benefit from staged approaches
- Inconsistent user experience when invoking Ralf
- Missed opportunities to leverage Ralf's planning capabilities
- Lack of delivery options (email) for research results

Users expect that saying "use Ralf" will consistently invoke the full planning and execution pipeline, not occasionally skip to investigator.

## Affected Components

- `ralf_manager.py` - Ralf orchestration and session management
- `ralf_spawn.py` - Ralf session initialization and routing
- `research_manager.py` - Research task handling and routing
- `telegram/` - Telegram integration for "use Ralf" command parsing
- `delivery/` - Email delivery integration for research results

## Suggested Implementation

### 1. Fix Ralf Invocation Routing
Modify the intent parsing and routing logic to consistently route complex research tasks to Ralf's planning pipeline:
- Enhance intent detection to identify complex research tasks (multi-step, multi-source, synthesis required)
- Always route these tasks through Ralf's full planning pipeline rather than direct investigator invocation
- Add heuristics to distinguish between quick one-off queries (OK for direct investigator) vs complex research (needs full Ralf)

### 2. Implement Stage Planning
Ensure Ralf breaks complex tasks into stages:
- Stage 1: Task decomposition and information needs identification
- Stage 2: Source planning and search strategy formulation  
- Stage 3: Execution of research (investigator calls per stage)
- Stage 4: Synthesis and result compilation
- Stage 5: Delivery (email with formatted results)

### 3. Add Email Delivery for Research Results
Implement email delivery option for Ralf research completion:
- Add email parameter to Ralf session configuration
- Format research results into structured email (HTML with sections, sources, summary)
- Integrate with existing `gmail_manager.py` for delivery
- Add confirmation message indicating email was sent

### 4. Improve User Feedback
Add clearer user feedback throughout Ralf execution:
- Confirm Ralf mode activation and planned stages
- Update progress as stages complete
- Provide final summary with delivery confirmation

### 5. Configuration Options
Add configuration for Ralf behavior:
- Always use full pipeline for research tasks
- Email delivery enabled/disabled default
- Stage timeout and retry configuration
- Research result formatting preferences

## Acceptance Criteria

1. **Complex Research Routing**: When a user requests complex research (e.g., "use Ralf to investigate X with multiple aspects"), the system consistently uses the full Ralf planning pipeline
2. **Stage Planning**: Ralf breaks down the task into 3-5 logical stages and presents the plan before execution
3. **Sequential Execution**: Stages execute in sequence with progress updates between each stage
4. **Email Delivery**: Research results are emailed to the user upon completion, with properly formatted HTML including sources, findings, and synthesis
5. **User Feedback**: User receives clear confirmation that Ralf is handling the task, sees the planned stages, gets progress updates, and receives delivery confirmation
6. **Backward Compatibility**: Simple queries can still use direct investigator when appropriate (configurable threshold)
7. **Configuration**: Admin can configure email delivery defaults and Ralf routing thresholds via config files

## Testing Considerations

- Test various research task complexities to ensure proper routing
- Verify email delivery works with different research result types
- Test stage planning with different research domains
- Verify progress updates are timely and informative
- Test error handling and recovery during stage execution