# Security policy

## Supported scope

Decompute is currently an experimental, local-development prototype. It is supported only on trusted localhost or trusted private-network deployments on macOS/Apple Silicon.

The coordinator and worker HTTP APIs do **not** provide authentication, authorization, TLS, rate limiting, or tenant isolation. Do not expose them directly to the public internet, a shared untrusted LAN, or a reverse proxy without adding those controls yourself.

Workers execute model inference only; they do not execute model-proposed tools. API clients and harnesses remain responsible for any tool execution.

## Reporting a vulnerability

Please do not file public issues for a suspected vulnerability. Email the repository maintainer privately with:

- a clear description of the impact;
- reproduction steps or a proof of concept;
- the affected revision and configuration; and
- any suggested mitigation.

You should receive an acknowledgement within seven days. Until a dedicated security contact is published, use the email address listed on the repository owner’s GitHub profile.
