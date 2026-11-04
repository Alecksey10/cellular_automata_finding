"""Живой просмотр two_rings_rot в терминале (общий стандарт visualize.live)."""

from __future__ import annotations

import sys

from visualize.live import main_for

from . import build_experiment

SCHEME = "age"

if __name__ == "__main__":
    raise SystemExit(main_for(build_experiment(), scheme=SCHEME, argv=sys.argv[1:]))
