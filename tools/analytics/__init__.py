"""habitat-gs benchmark analytics.

Contains:
- ``generate_report``  : cross-session report generator (CLI tool)
- ``nav_dashboard``    : web dashboard (CLI tool)
- ``session_stats``    : session-level stats collector (added in Phase 1 PR 2)
- ``trace_writer``     : per-round / per-event trace appenders (added in Phase 1 PR 2)

The top-level scripts (generate_report, nav_dashboard) were historically
run directly as scripts. Phase 1 adds proper ``__init__.py`` so other
subprocesses (e.g., nav_agent) can ``from analytics.session_stats import
collect_session_stats`` without needing to sys.path-manipulate.
"""
