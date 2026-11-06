"""Shared pytest setup.

Keep test runs out of Langfuse: the app loads .env (which may hold real
LANGFUSE_* keys), so tracing is switched off before any client starts. Set
LANGFUSE_TRACING_ENABLED=true in the environment to trace a test run anyway.
"""

import os

os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
