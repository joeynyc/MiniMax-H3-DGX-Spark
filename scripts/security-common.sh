#!/usr/bin/env bash

h3_validate_network_security() {
  local bind_host="$1"
  local allow_remote="$2"
  local api_key="$3"

  if [[ ! "$bind_host" =~ ^[A-Za-z0-9.:-]+$ ]]; then
    printf 'H3_BIND_HOST contains unsupported characters\n' >&2
    return 1
  fi

  if [[ -n "$api_key" && ! "$api_key" =~ ^[A-Za-z0-9._~-]{32,}$ ]]; then
    printf 'H3_API_KEY must contain at least 32 safe characters when configured\n' >&2
    return 1
  fi

  case "$bind_host" in
    127.0.0.1|localhost|::1)
      return 0
      ;;
  esac

  if [[ "$allow_remote" != "true" ]]; then
    printf 'remote bind refused; read SECURITY.md and set H3_ALLOW_REMOTE_API=true\n' >&2
    return 1
  fi

  if [[ -z "$api_key" ]]; then
    printf 'remote bind requires H3_API_KEY with at least 32 safe characters\n' >&2
    return 1
  fi
}
