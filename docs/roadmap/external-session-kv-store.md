# Externalized Session KV Store

## Summary

Decompute currently keeps cached llama.cpp contexts inside the worker process
and uses coordinator affinity to prefer the worker holding a session. Add a
future, optional store for serialized session/KV state so a session can be
evicted from an active context and restored later.

This is a resilience and resource-management feature, not an encryption
feature. KV snapshots remain sensitive plaintext-derived inference state.

## Why this may be useful

- Worker restart recovery without immediately losing every warm session.
- Lower RAM pressure when the number of sessions exceeds the number of live
  llama.cpp contexts.
- A path toward routing a session to another compatible worker after a failure
  or during capacity changes.
- Explicit lifecycle and quota management for session state.

For the current single-host deployment, a live worker-local context is expected
to be faster than serializing and restoring state. Externalization should be
opt-in and justified by measured memory, restart, or scheduling requirements.

## Model

The store holds serialized llama.cpp state for one session. It is not a chat
history database and must not be treated as a generic message store.

The expected lifecycle is:

1. Identify a session using the opaque client session UUID and requested model.
2. Validate the stored model/runtime/template compatibility metadata.
3. Restore the serialized KV state into a worker context, if available.
4. Validate the new tokenized prompt against the cached token prefix.
5. Decode only the uncached suffix and generate the response.
6. Snapshot the updated state back to the store after successful inference.
7. Discard the state on cancellation, inference failure, incompatible prompt,
   context overflow, or explicit invalidation.

The store must support bounded size, idle expiration, reset/delete, and
exclusive access for a session while a request is using its state.

## Proposed backends

### RAM

An in-process byte buffer is the default external backend. It reduces the
number of live contexts while avoiding disk I/O, but it does not survive a
process restart and is not shared between workers.

### Local disk

Persist one snapshot per session under a configured directory. This trades
latency and disk space for restart recovery and lower resident memory. Files
must be removed on TTL expiry, explicit deletion, model unload, or quota
eviction. Crash cleanup must be safe and bounded.

### Shared storage (future)

A network or shared-filesystem backend could allow compatible workers to
restore a session without coordinator affinity. This requires a clear locking
protocol, atomic writes, model/version namespacing, bandwidth limits, timeout
behavior, and protection against stale writers. It should not be assumed that
local disk provides this capability.

## Compatibility and safety requirements

Every snapshot must be namespaced or tagged with at least:

- model ID and model manifest/checksum;
- tokenizer and chat-template identity;
- runtime/serialization format version;
- context configuration relevant to restoration;
- token history or an equivalent prefix-validation record.

A mismatch must produce a cache miss and rebuild the context. It must never be
silently reused as if it were valid.

Snapshots contain sensitive model-derived state. Any disk or shared backend
must have restrictive permissions, bounded retention, and documented handling
for encryption at rest and in transit. Session UUIDs remain opaque routing
identifiers, not authentication credentials or encryption keys.

## API direction

Keep the store behind the worker/SDK runtime boundary. The coordinator should
continue to see only model ID and opaque session UUID; it should not inspect or
transport KV bytes.

The runtime-facing abstraction should provide operations equivalent to:

- load a committed snapshot for a session;
- prepare writable storage for a new snapshot;
- commit the new snapshot atomically;
- reset/delete a session;
- report size and lifecycle metadata;
- close the store cleanly.

The implementation must preserve the current no-session behavior: clients
without the session header continue to work and do not create cache entries.

## Scheduling relationship

External storage reduces, but does not eliminate, the value of affinity.
Affinity avoids snapshot load latency and should remain the preferred path when
the bound worker is healthy and has capacity. If the worker is unavailable,
the coordinator may select another compatible worker; that worker can restore
the snapshot if the backend is shared and the compatibility checks pass.

## Observability

Add privacy-safe counters and structured logs for:

- snapshot load hit/miss;
- restore failure by reason category;
- snapshot commit success/failure;
- bytes loaded, written, and evicted;
- session expiry, quota eviction, and invalidation;
- restore latency and snapshot I/O latency.

Never log session UUIDs, prompt text, token values, serialized KV bytes, or
authentication material.

## Acceptance criteria

- RAM backend passes load, commit, reset, TTL, quota, and concurrent-session
  ownership tests.
- Disk backend survives a worker restart and rejects incompatible snapshots.
- Cancellation and inference failures cannot commit partially updated state.
- Corrupt, truncated, stale, or incompatible snapshots become misses rather
  than inference correctness failures.
- Shared-backend behavior, if implemented, has atomic write and stale-writer
  tests.
- Benchmarks show when externalization is beneficial compared with a live
  context, including snapshot size, restore latency, and time-to-first-token.
- Documentation clearly states that externalization does not provide
  confidentiality from a privileged host process.

## Recommended sequencing

1. Measure the existing live-cache memory and latency behavior.
2. Introduce the runtime store trait with the in-process RAM implementation.
3. Add explicit snapshot compatibility metadata and failure handling.
4. Add local-disk persistence and restart-recovery tests.
5. Consider shared storage only after measuring the operational need and
   defining locking, security, and bandwidth behavior.
