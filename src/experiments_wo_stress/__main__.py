"""Allow ``python -m experiments_wo_stress`` alongside the ``ews`` command."""

from .cli import main

raise SystemExit(main())
