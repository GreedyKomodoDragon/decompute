# Roadmap Documentation Guide

This guide explains how agents and contributors should use the roadmap
documents in this repository.

## Document structure

- [`roadmap.md`](roadmap.md) is the index and triage list.
- [`roadmap/`](roadmap/) contains one detailed document per roadmap item.
- An item listed under **Unscheduled tasks** is planned work, not an
  implementation commitment.

## How to add or update work

When a new future task is identified, add a short entry to the appropriate
section of `roadmap.md` and link it to a detailed document in `roadmap/`.
Keep the index useful for scanning: include the goal, why it matters, and the
link, but move design detail into the linked document.

Detailed roadmap documents should describe the problem, intended behavior,
tradeoffs, interfaces, compatibility and privacy constraints, observability,
acceptance criteria, and recommended sequencing. Do not present an unscheduled
item as already implemented.

## Agent workflow

Before implementing a roadmap item:

1. Read its index entry and full detailed document.
2. Inspect the current code and tests; the roadmap is intent, not a substitute
   for repository truth.
3. Resolve any conflict in favor of explicit user instructions and current
   safety constraints.
4. Update the detailed document if implementation discovery changes the design.
5. Mark status or move the item only when the user explicitly requests roadmap
   maintenance or the repository's agreed project process defines that action.

When implementing, preserve the roadmap's stated out-of-scope boundaries and
record meaningful deviations in the detailed document or implementation notes.

## Privacy and security rule

Roadmap documents may describe sensitive state, but must not include real
session IDs, prompts, token contents, credentials, serialized KV data, or
machine-specific secrets. Treat session identifiers as opaque metadata and
state clearly when a feature is not encryption or authentication.
