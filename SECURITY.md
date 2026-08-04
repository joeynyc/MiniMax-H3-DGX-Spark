# Security policy

Do not open a public issue containing credentials, private infrastructure
addresses, model-access tokens, or sensitive generated content.

## Network defaults

The default configuration binds vLLM-Omni to `127.0.0.1`. It is reachable only
from the DGX Spark itself. A non-loopback bind fails in both preflight and the
container entrypoint unless all of the following are true:

- `H3_ALLOW_REMOTE_API=true` explicitly acknowledges remote access;
- `H3_API_KEY` contains at least 32 safe characters; and
- `.env` is not readable by group or other users.

Generate a unique key with `openssl rand -hex 32`, store it only in the ignored
`.env` file, and run `chmod 600 .env`. vLLM checks that bearer token on its
OpenAI-compatible routes. The included status and smoke scripts automatically
send the key when it is configured.

API-key middleware is not a firewall, TLS endpoint, or rate limiter, and some
non-OpenAI routes may remain unauthenticated. For access beyond one trusted
machine, restrict the port with host firewall and Tailscale ACL rules or place
an authenticated TLS reverse proxy in front of it. Never port-forward this
service directly to the public internet.

## Runtime trust

The launcher uses `--trust-remote-code` for MiniMax H3. Only mount a checkpoint
obtained from the authoritative MiniMax source and verify what you downloaded.
Remote model code runs inside a GPU-enabled container with host networking,
host IPC, a writable Hugging Face cache, and a writable output directory. The
checkpoint itself is mounted read-only, the container is not privileged, and
the base image is pinned by digest, but this is not a sandbox for untrusted
model repositories.

Treat prompts and generated output as sensitive. The project deliberately
keeps `.env`, model weights, caches, logs, and generated media out of Git.

The benchmark and quality scripts write request metadata, hashes, logs,
decoded inspection frames, and media under ignored `output/`. Review that
directory separately before sharing diagnostics. Never stage it, and replace
private paths or endpoint details with generic placeholders in public reports.

The Compose service deliberately has no restart policy and does not use
privileged mode. Review all container, model, and model-license changes before
upgrading the pinned base-image digest.
