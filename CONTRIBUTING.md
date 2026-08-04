# Contributing

Small, evidence-backed improvements are welcome.

Before opening a pull request:

1. Keep model weights, generated media, credentials, and private endpoints out
   of the repository.
2. Run `make audit`, `make provenance`, and `make test`. After building on
   ARM64, also run `make profile-test`.
3. Run `bash -n start-fp8.sh scripts/*.sh` and validate `fp8-quant.json`.
4. State the exact GPU architecture, image digest, command, and observed result
   for runtime changes.
5. Mark modifications to third-party Apache-2.0 source clearly.
6. Do not generalize one-machine measurements into vendor or upstream claims.

Runtime pull requests should explain the failure being addressed, why the
change is narrower than alternatives, and whether a cold start plus real
audio-video request passed.
