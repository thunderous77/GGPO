#!/usr/bin/env bash
# Audit the Git index for host-specific values committed into fork-owned files.
#
# This script is the single source of truth for the pattern and the scope. It
# deliberately lives outside AUDIT_SCOPE: a file that spells the pattern out
# would match itself, and an audit that always reports one hit trains people to
# ignore hits. Keep it that way — if scripts/ is ever added to AUDIT_SCOPE, the
# self-excluding pathspec below is what keeps this file from matching itself.
#
# Usage:
#   scripts/audit_host_paths.sh        # exit 0 when clean, 1 and print the leaks
#   scripts/audit_host_paths.sh --print-pattern
#
# Three elements are load-bearing; none may be dropped:
#   --cached      audit the index, not the working tree. A file marked
#                 skip-worktree (git ls-files -v | grep '^S') keeps a local
#                 version on disk that git never commits, so a working-tree
#                 grep reports clean while the committed version still carries
#                 the host value.
#   AUDIT_SCOPE   restrict to paths this fork owns. An unscoped pattern matches
#                 base64 blobs in third_party/ notebooks and buries the real
#                 hit in megabytes of noise.
#   -I            skip binary files.

set -uo pipefail

# Home directories are matched by shape, not by one known username: a path
# leaked from a second machine carries a different user. The trailing entries
# are per-site literals with no general shape (login nodes, scheduler
# partitions) — append new ones as hosts join, never remove the shape rule.
#
# Published dataset and software revisions are portable, not host paths.
readonly HOST_PATH_PATTERN='(/home|/u|/users)/[a-z][a-z0-9_-]+/|/work/nvme/|conda/envs/|gh-login|ghx4'

readonly AUDIT_SCOPE=(
    AGENTS.md
    AGENTS.override.example.md
    .agent/
    docs/contributing/
    examples/ggpo/
    diagnostics/
    verl/trainer/hybrid_rl/
    verl/trainer/offline_rl/
    tests/trainer/hybrid_rl/
    tests/trainer/offline_rl/
    tests/examples/
    ':!scripts/audit_host_paths.sh'
)

if [[ "${1:-}" == --print-pattern ]]; then
    printf '%s\n' "${HOST_PATH_PATTERN}"
    exit 0
fi

cd "$(git rev-parse --show-toplevel)" || exit 2

if git grep --cached -nIE "${HOST_PATH_PATTERN}" -- "${AUDIT_SCOPE[@]}"; then
    printf '\nHost-specific values are committed in the files above.\n' >&2
    printf 'Take them as required parameters that fail closed when unset; put\n' >&2
    printf 'the values in the host-local AGENTS.override.md instead.\n' >&2
    exit 1
fi

printf 'host-path audit clean\n'
