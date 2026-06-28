---
title: "YouTube Research Email: Fix Link Validation and Per-Exercise-Split Logic"
date: "2026-06-11"
priority: "High"
mode: feature
---

# YouTube Research Email: Fix Link Validation and Per-Exercise-Split Logic

## Description

The email generation pipeline for YouTube research results has two critical issues that
degrade user experience when sending analysis emails:

1. **YouTube links are sometimes incorrect or broken** — the link-generation stage
   does not validate URLs before embedding them in email bodies, so malformed, stale, or
   mis-constructed YouTube links reach the recipient.

2. **Email splitting aggregates multiple exercise series per email** — the intended
   behavior is that each exercise point/series receives its own standalone email with
   the correct YouTube links. Currently the splitting logic groups several series into
   a single message, making results harder to digest and reference.

## Motivation

Users rely on these analysis emails to quickly access curated YouTube content. Broken
links waste time and erode trust. Over-packed emails defeat the purpose of a digestible,
point-by-point research summary. Both issues make the feature unreliable in production.

## Affected Components

| Component | Role | What needs to change |
|-----------|------|----------------------|
| Email generation / composition module | Builds email body, embeds YouTube links | (a) Add URL validation (schema check + optional fetch/head check) before embedding a YouTube link; reject or flag invalid links. (b) Refactor split logic so each exercise series → one email. |
| YouTube research session processor | Produces the list of exercise points with associated YouTube links | Ensure links produced here are canonical (`https://www.youtube.com/watch?v=...` or short `youtu.be/...`), trim query noise, and attach a checksum or ID so downstream validation can verify them. |
| Email splitting / batching logic | Decides how to distribute content across outbound messages | Replace any threshold-based or count-based grouping with a 1:1 mapping: one exercise entry → one email. Remove/replace any "pack remaining items" fallback. |
| Telegram / user-facing notification | Tells the user what was sent | Update summary text to mention "N emails, one per exercise series" so the user has the right expectation. |

## Suggested Implementation

### 1. YouTube link validation layer

```
# Pseudocode
def validate_youtube_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    valid_domains = {"www.youtube.com", "youtube.com", "youtu.be"}
    if parsed.netloc not in valid_domains:
        return False
    # For youtu.be expect /<video_id> path; for youtube.com expect ?v=<id> or /v/<id>
    if parsed.netloc == "youtu.be":
        return len(parsed.path.strip("/")) >= 11
    video_id = parse_qs(parsed.query).get("v", [""])[0] or parse_path_id(parsed.path)
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id))
```

- Insert this check at the email composition stage. Skip or quarantine links that fail.
- Optionally add a lightweight HEAD request to the YouTube URL to confirm reachability
  (with a short timeout and fallback if YouTube rate-limits).

### 2. Per-exercise email splitting

```
# Before (broken)
emails = batch_entries(all_exercises, max_per_email=3)

# After (correct)
emails = [build_email(exercise) for exercise in all_exercises]
```

- Locate the batching/splitting function (likely named `split`, `batch`, `chunk`,
  or similar in the email module). Replace the grouping logic with per-item emission.
- Ensure the email subject line includes the exercise name so the inbox entry is informative.
- Ensure each email body is self-contained (standalone intro, links, summary — no
  cross-reference to sibling emails needed).

### 3. Integration with session processor

- The session that produces research results should output a clean, flat list of
  exercise entries, each with a verified link array. If the session itself generates
  links from video IDs or search results, validate at the source too (defense in depth).

## Acceptance Criteria

- [ ] Every YouTube link in an outbound email passes `validate_youtube_url()` before
      the email is sent.
- [ ] Invalid or missing links are either excluded with a warning in the email body
      or trigger a retry with the session processor to regenerate the link.
- [ ] Each exercise series produces exactly one email.
- [ ] Sending 5 exercise series results in 5 separate emails, each with valid links.
- [ ] No email contains more than one exercise series worth of content.
- [ ] User-facing notification (Telegram) correctly reports the number of emails sent
      and confirms per-exercise splitting.
- [ ] Existing non-YouTube email flows remain unaffected (no regression).
