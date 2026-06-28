#Feature Request: Proper Ralf Agent Integration as Default for Complex Multi-step Research

**Date:** 2026-06-11
**Priority:** High

## Description
Add proper Ralf agent as the default for complex multi-step research tasks instead of directly invoking the investigator agent. Ralf should decompose user requests into discrete stages, plan execution order, coordinate multiple tools/agents, and deliver final results including optional email delivery.

## Motivation
Current behavior (observed in session ses_14a48ee15ffeeBblmb5eF0ET5p): when a user asks for research, the orchestrator attempts to launch support/master_investigator directly with no stage planning, no multi-step strategy, and no email delivery capability. The agent spent multiple turns fighting CLI syntax and never completed the actual research task.

## Affected Components
- scripts/opencode_manager.py: Agent spawning and routing
- Persona/orchestrator agent: Needs to recognize use Ralf keywords and route correctly
- support/master_investigator: Should become a subprocess called by Ralf
- Ralf agent: New or enhanced agent implementing the planning pipeline
- Email sending mechanism: Final delivery step

## Suggested Implementation
1. Create or configure the Ralf agent with stage-based planning
2. Add intent recognition for use Ralf in orchestrator agent
3. Integrate email delivery into Ralf final stage
4. Fix opencode_manager.py CLI pattern mismatch between security rules and actual CLI
5. Add configurable Ralf default mode for research requests

## Acceptance Criteria
- Ralf produces a numbered stage plan before executing
- Ralf coordinates investigator for actual research stages
- Ralf compiles a final synthesized report from all stages
- Email delivery works when requested
- Simple one-fact queries still work quickly without Ralf overhead
