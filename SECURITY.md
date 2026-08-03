# Security policy

Do not open a public issue containing credentials, private infrastructure
addresses, model-access tokens, or sensitive generated content.

This repository exposes a network service with host networking. Bind or
firewall it according to your environment; the example assumes a trusted local
network and does not add authentication. Do not expose the endpoint directly to
the public internet.

The Compose service deliberately has no restart policy and does not use
privileged mode. Review all container, model, and model-license changes before
upgrading the pinned base-image digest.
