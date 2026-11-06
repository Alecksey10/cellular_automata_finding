#!/usr/bin/env bash
# Сборка статической галереи в отдельный каталог.
#
# Каталог сборки — не часть репозитория: по умолчанию это pages/_site,
# а в GitHub Actions сюда передают путь внутри $RUNNER_TEMP.
#
#   pages/build.sh                    # -> pages/_site
#   pages/build.sh /tmp/sweep-site    # -> /tmp/sweep-site
#   pages/build.sh /tmp/sweep-site --page-size 48 --title "sweep langton"
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"          # try_3_code
out="${1:-${OUT_DIR:-$here/_site}}"
if [ "$#" -gt 0 ]; then
    shift
fi

cd "$root"
exec python3 -m sweep export --runs "${RUNS_DIR:-runs}" --out "$out" "$@"
