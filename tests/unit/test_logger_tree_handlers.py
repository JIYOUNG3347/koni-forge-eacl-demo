"""Regression tests for agent.* / agents.* logger tree handler attachment.

Importing server.core.logging must attach a handler to each parent node of
the logger trees used in the codebase, so that child INFO logs (e.g.
`agent.restructure`, `agents.dispatcher`) emit through propagation instead
of being silently dropped by a handler-less root.
"""

import logging

import server.core.logging  # noqa: F401  — import triggers handler attachment


class TestLoggerTreeHandlerAttachment:
    """Parent nodes of agent.* / agents.* must have handlers after import."""

    def test_agent_parent_has_handler(self):
        """`agent` (singular tree used by specialists) must have ≥1 handler."""
        assert logging.getLogger("agent").handlers, (
            "agent.* tree has no parent handler — specialist INFO logs will "
            "be dropped (regression of PR fix/twist-b-agent-logger)"
        )

    def test_agents_parent_has_handler(self):
        """`agents` (plural tree used by dispatcher/foundation) must have ≥1 handler."""
        assert logging.getLogger("agents").handlers, (
            "agents.* tree has no parent handler — dispatcher/foundation INFO "
            "logs will be dropped (regression of PR fix/twist-b-agent-logger)"
        )

    def test_koni_forge_parent_has_handler(self):
        """Pre-existing `koni_forge` tree must still have a handler (no regression)."""
        assert logging.getLogger("koni_forge").handlers


class TestNoDoubleHandlers:
    """get_logger() guard must prevent duplicate handler attachment across imports."""

    def test_reimport_does_not_duplicate_handlers(self):
        """Calling get_logger('agent') again must not add another handler."""
        from server.core.logging import get_logger

        before = list(logging.getLogger("agent").handlers)
        get_logger("agent")
        after = list(logging.getLogger("agent").handlers)
        assert before == after, "get_logger() re-attached handlers — duplicate emission risk"
