#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0
report_failure() {
  printf 'public audit failed: %s\n' "$1" >&2
  fail=1
}

check_current_and_history() {
  local label="$1"
  local pattern="$2"
  local commit
  local file

  while IFS= read -r file; do
    [[ "$file" != scripts/public-audit.sh && -f "$file" ]] || continue
    if grep -Iq . "$file" && grep -Eq "$pattern" "$file"; then
      report_failure "$label found in publishable worktree files"
      break
    fi
  done < <(git ls-files --cached --others --exclude-standard)

  for commit in $(git rev-list --all); do
    if git grep -qI -E "$pattern" "$commit" -- ':!scripts/public-audit.sh'; then
      report_failure "$label found in reachable history"
      return
    fi
  done
}

check_history_filenames() {
  local commit
  for commit in $(git rev-list --all); do
    if git ls-tree -r --name-only "$commit" | grep -Eiq '\.(mp4|mov|mkv|avi|jpg|jpeg|png|webp|safetensors|bin|pt|pth|ckpt)$'; then
      report_failure 'generated media or model-like binary found in reachable history'
      return
    fi
  done
}

check_current_and_history 'host-specific absolute path' '/(home|Users)/[^/[:space:]]+'
check_current_and_history 'private or Tailscale/CGNAT IPv4 address' '(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3})'
check_current_and_history 'private hostname or local username' '(JoeyDGX|joeydgx|gx10(-3028)?|spark-db08|zerocool)'
check_current_and_history 'email address' '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}'
check_current_and_history 'credential value' '(-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|(api[_-]?key|password|secret)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9])'

if git ls-files --cached --others --exclude-standard | grep -Eiq '\.(mp4|mov|mkv|avi|jpg|jpeg|png|webp|safetensors|bin|pt|pth|ckpt)$'; then
  report_failure 'generated media or model-like binary is publishable'
fi
check_history_filenames

while IFS= read -r file; do
  [[ -f "$file" ]] || continue
  if (( $(stat -c '%s' "$file") > 5 * 1024 * 1024 )); then
    report_failure "publishable file larger than 5 MiB: $file"
  fi
done < <(git ls-files --cached --others --exclude-standard)

git diff --check || report_failure 'whitespace errors found'
git fsck --no-progress >/dev/null || report_failure 'git object check failed'

(( fail == 0 )) || exit 1
echo 'public audit passed'
