# Local SigNoz test stack

This directory uses SigNoz Foundry to render a supported Docker Compose
deployment. The generated Compose files are intentionally not committed;
Foundry owns them and may change their layout between releases.

## Start SigNoz

```bash
foundryctl cast -f docker/signoz/casting.yaml
```

SigNoz exposes its UI on `http://127.0.0.1:8080` and OTLP HTTP ingestion on
`http://127.0.0.1:4318`. OTLP gRPC is available on port `4317`.

To render without starting the stack:

```bash
foundryctl gauge -f docker/signoz/casting.yaml
foundryctl forge -f docker/signoz/casting.yaml
docker compose \
  -f pours/deployment/compose.yaml \
  -f docker/signoz/compose.override.yaml up -d
```

## Run Decompute against it

Start the coordinator normally, then start a worker with the OTLP endpoint:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
OTEL_SERVICE_NAME=decompute-worker-a \
just worker-a
```

The worker’s process-local cache logs and cumulative cache metrics are pushed
to SigNoz. Send a request through the coordinator, then open the SigNoz UI and
look for service `decompute-worker-a`. Cache counters use these metric names:

- `decompute.session_cache.hits`
- `decompute.session_cache.misses`
- `decompute.session_cache.evictions`
- `decompute.session_cache.expirations`
- `decompute.session_cache.invalidations`

When `--session-cache-capacity 0` is used, workers bypass both KV retention and
same-session serialization. A non-zero `--session-cache-max-bytes` reserves a
conservative estimate for each full-context KV allocation; it is intentionally
overestimated to avoid admitting more cached contexts than the budget allows.

For a second worker, use a different service name and port as usual. The
workers send telemetry directly to the forwarded host OTLP port; no
coordinator routing is involved.

## Stop and inspect

```bash
docker compose -f pours/deployment/compose.yaml -f docker/signoz/compose.override.yaml ps
docker compose -f pours/deployment/compose.yaml -f docker/signoz/compose.override.yaml logs -f
foundryctl destroy -f docker/signoz/casting.yaml
```

SigNoz requires roughly 4 GB of Docker memory. The generated stack and its
data volumes are local test state and can be removed with Foundry after the
test.
