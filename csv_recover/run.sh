#!/usr/bin/env bash
# Convenience entry points. Always invoke Python in isolated mode (-I) so the
# untrusted data directory cannot shadow standard modules.
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  fixtures)
    python3 -I tests/make_fixtures.py fixtures
    ;;
  test)
    python3 -I tests/test_recover.py
    ;;
  run)
    shift
    python3 -I csv_recover/recover.py "$@"
    ;;
  *)
    echo "usage: $0 {fixtures|test|run <input.json> [--json|-o report.json]}" >&2
    exit 2
    ;;
esac
