#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/security-common.sh"

strong_key='0123456789abcdef0123456789abcdef'

expect_accept() {
  h3_validate_network_security "$1" "$2" "$3" >/dev/null 2>&1 || {
    printf 'expected network policy to accept bind %s\n' "$1" >&2
    exit 1
  }
}

expect_refuse() {
  if h3_validate_network_security "$1" "$2" "$3" >/dev/null 2>&1; then
    printf 'expected network policy to refuse bind %s\n' "$1" >&2
    exit 1
  fi
}

expect_accept 127.0.0.1 false ''
expect_accept localhost false ''
expect_accept ::1 false ''
expect_refuse 127.0.0.1 false 'too-short'
expect_refuse 0.0.0.0 false ''
expect_refuse 0.0.0.0 true 'too-short'
expect_refuse 'bad bind' true "$strong_key"
expect_accept 0.0.0.0 true "$strong_key"
expect_accept spark.example true "$strong_key"

printf 'network security tests passed\n'
