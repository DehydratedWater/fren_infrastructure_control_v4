# Feature Request: Better YouTube Link Handling and Email Splitting

**Date:** 2026-06-14
**Priority:** High

## Description

Email generation for YouTube research results has two critical issues:

1. **Broken YouTube Links**: When sending analysis with YouTube links, the links are sometimes incorrect or broken
2. **Improper Email Splitting**: The splitting logic places multiple exercise series in a single email instead of creating one email per exercise series

## Motivation

Users request YouTube research analysis and expect:
- Correct, clickable YouTube video links
- One email per exercise series (not multiple series combined)
- Clear separation of content across emails for better readability

Currently, these failures frustrate users and reduce the utility of the research email delivery system.

## Affected Components

Based on the codebase structure, the following components are likely involved:

- **Email Sending**: backend/app/services/gmail_client.py - handles email composition and sending
- **YouTube Fetching**: scripts/youtube_fetcher.py - handles YouTube link processing
- **Research Management**: backend/app/agents/domains/research.py - manages research workflows
- **Email Splitting Logic**: Likely within research agents or delivery components

## Current Issues

### Issue 1: Broken/Incorrect YouTube Links
- YouTube links in generated emails may be malformed or incomplete
- Link validation likely missing before email composition
- URL encoding or formatting issues when processing research results

### Issue 2: Improper Email Splitting
- Email splitting logic doesn't properly detect boundaries between exercise series
- Multiple related research points are grouped together incorrectly
- Logic may rely on content length rather than semantic boundaries

## Suggested Implementation

### Fix 1: YouTube Link Validation
1. Add link validation in YouTube processing pipeline
2. Validate YouTube URLs follow standard format: https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID
3. Extract and verify video IDs exist in URLs
4. Add error logging for invalid links with research context
5. Provide fallback mechanism when links are invalid

### Fix 2: Semantic Email Splitting
1. Implement semantic boundary detection for exercise series
2. Detect common patterns indicating new series:
   - Exercise Series #X or similar numbered headers
   - Topic changes in research output
   - Distinct video groups with different themes
3. Create one email per semantic boundary rather than arbitrary character/line limits
4. Preserve context and ensure each email is self-contained with relevant links

### Code Locations to Modify

1. **YouTube Link Processor**: Add validation in scripts/youtube_fetcher.py or relevant YouTube processing module
2. **Email Composer**: Update email construction in backend/app/services/gmail_client.py
3. **Research Agent**: Modify splitting logic in backend/app/agents/domains/research.py or related research components
4. **Split Detection**: Implement new boundary detection algorithm (may require new module)

## Acceptance Criteria

- [ ] YouTube links in emails are always valid and clickable
- [ ] Each exercise series appears in a separate email
- [ ] No YouTube links are malformed or contain invalid video IDs
- [ ] Email splitting respects semantic boundaries of content
- [ ] Error logging captures any link validation failures for debugging
- [ ] User receives exactly N emails for N exercise series
- [ ] All YouTube video links work and point to correct videos