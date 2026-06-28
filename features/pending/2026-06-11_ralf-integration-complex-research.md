---
title: Ralf Integration as Default Orchestrator for Complex Multi-Step Research
date: 2026-06-11
priority: Medium-High

## Description
Currently, when a user requests to 'use Ralf' for complex research tasks, the system inconsistently falls back to running the investigator agent directly, bypassing Ralf's full planning pipeline. Ralf should serve as the primary orchestrator for multi-step research tasks, breaking work into stages, planning each stage, and executing via child agents (such as the investigator). The system should also support sending final results by email at the end of a research workflow.

## Motivation
**Problem:** Users say 'use Ralf' expecting a structured, multi-stage research process with planning, stage decomposition, and execution — but instead get a raw investigator run without any orchestration.

**Why it matters:**
- Ralf's value proposition is its planning pipeline: task decomposition -> stage planning -> sequential/parallel execution -> result aggregation
- Without this pipeline, users lose the benefit of structured, traceable, multi-step research
- Researchers and analysts expect email delivery of results for offline review and record-keeping
- The current inconsistency (sometimes Ralf pipeline, sometimes just investigator) erodes user trust in the system

## Affected Components
| Component | Role | Change Needed |
|-----------|------|---------------|
| Ralf agent | Primary orchestrator | Ensure it always runs full planning pipeline before dispatching child agents |
| Command routing | Maps user intents to agents | Add intent detection for complex research that explicitly routes to Ralf, not investigator |
| Investigator agent | Single-task researcher | Remains a leaf executor, invoked by Ralf stage planner, never standalone for complex tasks |
| Email delivery | Result notification | Add post-execution email reporting to Ralf pipeline |
| opencode_manager | Agent launcher | Ensure Ralf launches are tracked as orchestration sessions, not simple sessions |

## Suggested Implementation

### 1. Intent Routing Fix
Add a keyword/intent detection layer that distinguishes:
- Simple lookup -> investigator directly (fast, single-hop)
- Complex research -> Ralf orchestrator (multi-stage, planned execution)

Trigger words for Ralf: research, investigate thoroughly, deep dive, use Ralf, comprehensive analysis, compare and report

### 2. Ralf Planning Pipeline Enforcement
Ralf's system prompt and execution flow must guarantee:
1. Task analysis — parse user request, determine scope and domain
2. Stage decomposition — break into discrete, ordered stages (gather sources, analyze, cross-reference, synthesize)
3. Plan review — present or log the plan before execution
4. Stage execution — spawn investigator (or other child agents) per stage, collect results
5. Aggregation — merge partial results into a single report
6. Email delivery — send final report to configured email address(es)

### 3. Email Delivery Integration
- Add a --send-email flag or config option to Ralf
- After aggregation, generate an email-friendly summary
- Use the existing email infrastructure (based on prior sessions around email splitting/YouTube links, the system already has email capabilities)
- Include: subject with task slug, body with executive summary, attachment or link to full report

### 4. Session Tracking
- Ralf orchestration sessions should have a distinct session tree structure:
  - Root session = Ralf planning session
  - Child sessions = individual stage executions (investigator, etc.)
- This enables traceability: users can see the plan and each stage's output

## Acceptance Criteria
- [ ] User says 'use Ralf to research X' -> Ralf runs full planning pipeline (never bypasses directly to investigator)
- [ ] Ralf outputs a visible stage plan before execution begins
- [ ] Each stage completes, results are collected before the next stage starts
- [ ] Final aggregated report is produced
- [ ] User can opt in to receive the final report by email
- [ ] Simple queries ('what is X') still route to investigator directly without Ralf overhead
- [ ] Session tree shows Ralf root + child stage sessions for traceability