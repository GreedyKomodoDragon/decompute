# Decompute Roadmap

This document is the index for planned work. Completed work belongs in the
repository history and implementation documentation; this file tracks work
that has not yet been scheduled for implementation.

## Unscheduled tasks

### Worker session-cache hardening and telemetry

The first in-memory implementation now covers minimum-token admission,
bounded same-session ownership, prompt-boundary checkpoints, approximate byte
budgets, transactional publication, LRU/TTL lifecycle, and OTLP logs/metrics.
Future work should benchmark these defaults against real model/context sizes
and add richer runtime-specific byte accounting where llama.cpp exposes it.

The OTLP path is deliberately opt-in and content-blind. It exports cache
outcome categories and cumulative counters, never session IDs, prompts, or
token values. It does not change the confidentiality boundary: the coordinator
still handles plaintext requests and worker KV memory remains sensitive.

### Externalized session KV storage

Add an optional session-store abstraction that can move serialized llama.cpp
KV state out of a live worker context and restore it for a later request. The
default should remain the current worker-local live cache and coordinator
affinity because that is the lowest-latency path for the current deployment.

The feature is intended for concrete needs such as worker restart recovery,
reduced RAM pressure across many sessions, and eventually routing a session to
another compatible worker. It should not be implemented merely as a transcript
database: the stored payload is runtime KV/session state, not just messages.

Full design details, tradeoffs, compatibility requirements, and acceptance
criteria are in [Externalized Session KV Store](roadmap/external-session-kv-store.md).
