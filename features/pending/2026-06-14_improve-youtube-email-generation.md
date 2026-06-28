# Feature: Improve Email Generation for YouTube Research Results

**Date:** 2026-06-14

**Priority:** High

## Description

The email generation system for YouTube research results has two critical issues when handling split emails for exercise series analysis:

1. **YouTube Link Integrity:** YouTube links included in split emails are sometimes incorrect or broken, resulting in non-functional links when the user tries to access the referenced content.

2. **Email Splitting Logic:** The splitting logic incorrectly groups multiple exercise series into a single email instead of distributing each exercise series across separate emails. This defeats the purpose of splitting content for better readability and organization.

## Motivation

Users request analysis of YouTube exercise research with links split across multiple emails for better organization and readability. Currently, the system fails to:
- Generate correct, working YouTube links in each email
- Properly distribute exercise series content (each series should get its own email)
- Maintain link integrity when splitting content across multiple messages

This leads to frustration as users cannot access the referenced YouTube content and receive poorly organized email output.

## Affected Components

### Primary Components
- **`/home/dw/programing/fren_infrastructure_control_v4/scripts/render_and_send.py`** (13.9 KB)
  - Main email rendering and sending logic
  - Handles content splitting across multiple emails
  - Processes YouTube links in content

- **`/home/dw/programing/fren_infrastructure_control_v4/backend/app/tools/research/youtube_fetcher.py`** (176 bytes)
  - YouTube content fetching and link generation
  - Should validate and ensure link correctness

- **`/home/dw/programing/fren_infrastructure_control_v4/backend/app/tools/research/research_manager.py`** (179 bytes)
  - Research coordination and data structure management
  - Handles organization of research results including exercise series

### Secondary Components
- **`/home/dw/programing/fren_infrastructure_control_v4/backend/app/tools/comms/gmail_manager.py`** (164 bytes)
  - Gmail integration for email delivery
  - May be involved in email payload preparation

## Suggested Implementation

### Phase 1: Fix YouTube Link Integrity

1. **Add YouTube Link Validation**
   - Implement URL validation function in `youtube_fetcher.py` to verify YouTube link format
   - Check for required YouTube video ID and domain structure
   - Validate that links are properly formatted: `https://www.youtube.com/watch?v=[VIDEO_ID]` or `https://youtu.be/[VIDEO_ID]`

2. **Implement Link Preservation During Splitting**
   - Modify `render_and_send.py` to maintain a mapping between content sections and their associated YouTube links
   - Ensure that when content is split, each section retains its correct YouTube links
   - Add validation step before sending to verify all links are correctly formatted

3. **Add Fallback Link Generation**
   - If link appears broken, attempt to reconstruct from video ID
   - Add logging for broken links to aid debugging

### Phase 2: Fix Email Splitting Logic

1. **Improve Exercise Series Detection**
   - Implement regex or structured parsing to identify exercise series boundaries
   - Look for patterns like:
     - Series title headers
     - Numbering sequences (1., 2., 3., etc.)
     - Section separators
     - Exercise group indicators

2. **Implement Series-Aware Splitting**
   - Modify `render_and_send.py` to:
     - First parse the content into exercise series units
     - Count total series
     - Calculate optimal split points at series boundaries (never split within a series)
     - Generate one email per exercise series or group series appropriately

3. **Add Splitting Validation**
   - Ensure no email contains partial exercise series
   - Verify each email has proper YouTube links for its content
   - Add option for user to specify splitting granularity (by series, by count, etc.)

### Phase 3: Enhanced Error Handling

1. **Add Content Integrity Checks**
   - Validate that split emails have complete, coherent content
   - Verify YouTube links are properly associated with content
   - Check for orphaned links (links without context) and missing links (content without links)

2. **Improve Logging and Debugging**
   - Log splitting decisions and rationale
   - Track link generation and validation attempts
   - Provide clear error messages for validation failures

3. **Add Retry/Fallback Mechanisms**
   - If link validation fails, attempt alternative formats
   - If splitting produces uneven distribution, re-balance
   - Provide user feedback on any content modifications

## Acceptance Criteria

### YouTube Link Integrity
- [ ] All YouTube links in split emails are correctly formatted and functional
- [ ] Links are properly preserved when content is split across multiple emails
- [ ] No broken or incorrect YouTube links are generated
- [ ] Link validation passes for all generated emails
- [ ] Users can successfully click and open all YouTube links

### Email Splitting Logic
- [ ] Exercise series are not split across multiple emails
- [ ] Each email contains complete exercise series information
- [ ] Splitting respects exercise series boundaries
- [ ] Multiple exercise series are distributed across emails in a balanced manner
- [ ] Users receive one email per exercise series (or user-specified grouping)

### Integration and Quality
- [ ] Changes integrate seamlessly with existing email generation flow
- [ ] No regression in email delivery success rate
- [ ] Performance impact is minimal (splitting and validation complete in < 2 seconds for typical research results)
- [ ] Error messages are clear and actionable
- [ ] Logging provides sufficient detail for debugging

### User Experience
- [ ] Users can request split emails with confidence that content will be properly organized
- [ ] YouTube links are always accessible and correct
- [ ] Exercise series remain coherent and complete within each email
- [ ] System handles edge cases (single series, large series, mixed content)
- [ ] Polish language support maintained (based on user's request language)

## Code Changes Required

### 1. `render_and_send.py` - Email Splitting Logic
```python
# Add exercise series detection
def detect_exercise_series(content: str) -> List[dict]:
    """Parse content into exercise series units with associated links."""
    pass

# Modify splitting to respect series boundaries
def split_content_by_series(content: str, max_emails: int = None) -> List[str]:
    """Split content at exercise series boundaries."""
    pass

# Add link validation and preservation
def validate_youtube_links(content: str) -> bool:
    """Validate all YouTube links in content are correctly formatted."""
    pass
```

### 2. `youtube_fetcher.py` - Link Generation
```python
# Add URL validation
def validate_youtube_url(url: str) -> bool:
    """Validate YouTube URL format and extract video ID."""
    pass

# Add link normalization
def normalize_youtube_url(url: str) -> str:
    """Convert any YouTube URL format to standard format."""
    pass
```

### 3. `research_manager.py` - Data Structure
```python
# Add structured series representation
class ExerciseSeries:
    """Represents a single exercise series with metadata and links."""
    def __init__(self, title: str, content: str, youtube_links: List[str]):
        self.title = title
        self.content = content
        self.youtube_links = youtube_links
```

## Testing Recommendations

1. **Unit Tests**
   - Test YouTube URL validation with various formats
   - Test exercise series detection with different content patterns
   - Test splitting logic with various series counts and sizes

2. **Integration Tests**
   - Test end-to-end email generation with YouTube research results
   - Test splitting with mixed content (series, descriptions, links)
   - Test edge cases (single series, very long series, no links)

3. **Manual Testing**
   - Generate split emails for actual YouTube exercise research
   - Verify all links work in browser
   - Confirm series boundaries are respected in email content

## Notes

- The user's request includes Polish text: "Proszę napisać raport o tym co trzeba zmienić w kodzie" (Please write a report about what needs to be changed in the code)
- This suggests the system should maintain multi-language support in error messages and validation feedback
- Prioritize the splitting logic fix as it affects content organization
- YouTube link integrity should be addressed simultaneously as both issues impact user experience