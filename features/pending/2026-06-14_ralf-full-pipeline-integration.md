
# Feature Report: Full Ralf Pipeline Integration for Complex Research

## Date
2025-06-14

## Priority
High

## Description
Add proper Ralf integration as the default research orchestration mechanism for complex multi-step research tasks. When users invoke "use Ralf" or request complex research, the system should use the full Ralf planning pipeline (task decomposition into stages, planning, and execution) instead of falling back to running investigator in isolation.

## Motivation
Currently, when users request complex research with "use Ralf", the system sometimes only runs investigator without the full Ralf orchestration capabilities. This limits the ability to handle multi-stage research tasks that require:
- Task decomposition into logical stages
- Planning and sequencing of stages
- Proper context handoff between stages
- Comprehensive result aggregation

Additionally, there is a need to send completed research results by email for archival and offline review.

## Affected Components
- `ralf_spawn.py` - Ralf task spawning and orchestration
- `ralf_manager.py` - Ralf lifecycle management
- `investigator` agent - Research execution agent
- `intent_inference.py` - May need updates to better detect complex research requests
- `gmail_manager.py` - Email result delivery

## Suggested Implementation

### 1. Ralf Pipeline Integration
- Modify `ralf_spawn.py` to ensure full pipeline is invoked when Ralf is requested:
  - Stage 1: Task analysis and decomposition
  - Stage 2: Planning and sequencing
  - Stage 3: Execution (coordinating investigator, web_search, and other tools)
  - Stage 4: Result aggregation and formatting
  - Stage 5: Optional email delivery

- Add a `--full-pipeline` flag to Ralf spawn commands
- Update intent inference to prefer Ralf for multi-step research indicators:
  - Keywords: "complex", "multi-step", "stages", "comprehensive"
  - Task length indicators (> 3 related queries)
  - Cross-domain research requests

### 2. Email Result Delivery
- Add `--email-results` flag to Ralf spawn
- Integrate with `gmail_manager.py` to send formatted research summaries
- Email should include:
  - Task description and stages executed
  - Key findings per stage
  - Final synthesized summary
  - Links to detailed logs

### 3. Research Session Tracking
- Ensure Ralf sessions are properly logged in execution ledger
- Track stage transitions and time spent per stage
- Enable session inspection for debugging complex tasks

## Acceptance Criteria

1. When user says "use Ralf" with a complex research task, full pipeline runs (not just investigator)
2. Task decomposition produces 3+ stages for complex tasks
3. Results can be emailed with proper formatting
4. Execution ledger shows full Ralf session with stage breakdown
5. Simple single-query tasks still work efficiently (no unnecessary overhead)
6. Session inspector can trace Ralf pipeline execution
