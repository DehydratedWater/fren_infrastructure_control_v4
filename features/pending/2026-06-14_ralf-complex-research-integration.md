# Feature: Ralf Complex Research Integration

**Date:** 2026-06-14
**Priority:** High

## Description

Ralf should be the default research agent for complex multi-step research tasks, providing proper planning, staging, and email delivery capabilities.

## Motivation

Currently when users invoke "use Ralf" for complex research tasks, the system sometimes runs just the investigator agent without utilizing the full Ralf planning pipeline. This results in:

1. Incomplete research execution - tasks aren't properly broken down into stages
2. Missing planning phase - no systematic approach to complex multi-step research
3. No email delivery - results aren't automatically sent via email at completion
4. Inconsistent behavior - "use Ralf" sometimes works, sometimes doesn't

Users need reliable, comprehensive research capability that leverages Ralf's full capabilities for planning, staging execution, and automated email delivery of results.

## Affected Components

- `scripts/ralf_spawn.py` - Ralf initialization and pipeline orchestration
- `scripts/ralf_manager.py` - Ralf lifecycle management
- `scripts/research_manager.py` - Research task routing and agent selection
- `scripts/investigator.py` - Investigator agent (currently being called directly instead of via Ralf)
- `scripts/gmail_manager.py` - Email delivery for research results
- Agent routing logic that determines when to use Ralf vs investigator

## Suggested Implementation

### Phase 1: Ralf Pipeline Enhancement

1. **Planning Stage**
   - Implement task decomposition that breaks complex research into logical stages
   - Add stage dependency tracking (what stages must complete before others start)
   - Create stage execution queue with proper ordering

2. **Execution Staging**
   - Each stage should have: objective, tasks, success criteria, outputs
   - Implement checkpoint/resume capability for long-running research
   - Add stage progress tracking and reporting

3. **Email Integration**
   - Add email delivery configuration to Ralf pipeline
   - Implement result aggregation from all stages
   - Create comprehensive email template for research findings
   - Add email delivery confirmation and retry logic

### Phase 2: Routing Logic Updates

1. **Task Complexity Detection**
   - Add complexity heuristics: multiple sub-tasks, multi-step process, cross-domain research
   - Implement routing rules: complex tasks → Ralf, simple queries → investigator
   - Add fallback: if Ralf fails, degrade to investigator with warning

2. **"Use Ralf" Command Handling**
   - Always invoke full Ralf pipeline when explicitly requested
   - Provide clear feedback when Ralf is being used
   - Show pipeline stages and progress to user

### Phase 3: Integration Points

1. **Research Manager Integration**
   - Update `research_manager.py` to prefer Ralf for complex tasks
   - Add task complexity scoring before agent selection
   - Implement graceful degradation for Ralf failures

2. **Email Manager Integration**
   - Extend `gmail_manager.py` to support research result emails
   - Add email template for research summaries
   - Implement attachment handling for research artifacts

### Code Structure

```python
# ralf_spawn.py enhancements
class RalfResearchPipeline:
    def __init__(self, task, email_config):
        self.task = task
        self.stages = []
        self.email_config = email_config
    
    def plan_stages(self):
        """Break task into logical stages"""
        # Implementation
    
    def execute_stages(self):
        """Execute stages in dependency order"""
        # Implementation
    
    def deliver_results(self):
        """Send email with aggregated results"""
        # Implementation
```

## Acceptance Criteria

1. **Planning**
   - [ ] Ralf breaks complex research tasks into at least 2 stages
   - [ ] Stages have clear objectives and success criteria
   - [ ] Stage dependencies are properly tracked

2. **Execution**
   - [ ] Stages execute in correct order respecting dependencies
   - [ ] Progress is reported after each stage completion
   - [ ] Failed stages can be retried individually

3. **Email Delivery**
   - [ ] Research results are sent via email on completion
   - [ ] Email includes all stage outputs and summary
   - [ ] Email delivery confirmation is provided

4. **Routing**
   - [ ] "Use Ralf" always invokes full pipeline (not just investigator)
   - [ ] Complex tasks automatically route to Ralf
   - [ ] Simple queries still use investigator for speed

5. **UX**
   - [ ] Users see pipeline stages and progress
   - [ ] Clear feedback when Ralf vs investigator is used
   - [ ] Email is sent to user-configured address

## Testing

- Unit tests for stage planning and dependency resolution
- Integration tests for Ralf → investigator → email flow
- E2E tests for "use Ralf" command on complex tasks
- Manual testing with real research scenarios

## Risks

- Ralf pipeline complexity may increase failure surface
- Email delivery reliability dependent on external service
- May need rate limiting for stage execution
- Potential for duplicate research if stages overlap

## Success Metrics

- Ralf invoked successfully for 90%+ of complex research tasks
- Email delivery success rate >95%
- User satisfaction with research completeness increases
- Reduction in "use Ralf" failing or using investigator instead