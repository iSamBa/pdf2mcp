"""E2E tests for MCP server transports.

Tests real MCP client-server communication over HTTP and stdio
with real LanceDB data and mocked embeddings.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(60)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Server not ready on {host}:{port}")


def _run_server_in_thread(server: Any) -> None:
    """Run an async uvicorn server in a new event loop on a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())
    loop.close()


async def _with_session(url: str, coro_fn: Any) -> Any:
    """Connect to an HTTP MCP server, run a coroutine, return result."""
    async with (
        streamable_http_client(url) as (read_stream, write_stream, _),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        return await coro_fn(session)


# ---------------------------------------------------------------------------
# HTTP transport fixture (session-scoped — single server for all HTTP tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mcp_http_url(ingested_project_session: Path) -> str:
    """Start HTTP server once for the session, yield URL, shut down at end."""
    import uvicorn

    # Set env vars so the server can find the data
    os.environ["PDF2MCP_DATA_DIR"] = str(ingested_project_session / "data")
    os.environ["PDF2MCP_DOCS_DIR"] = str(ingested_project_session / "docs")
    os.environ["OPENAI_API_KEY"] = "sk-fake-e2e-key"

    from pdf2mcp.config import get_settings
    from pdf2mcp.store import _cached_connect, invalidate_table_cache

    get_settings.cache_clear()
    _cached_connect.cache_clear()
    invalidate_table_cache()

    from pdf2mcp.server import mcp as mcp_app

    port = _find_free_port()
    host = "127.0.0.1"

    app = mcp_app.streamable_http_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=_run_server_in_thread, args=(server,), daemon=True)
    thread.start()
    _wait_for_port(host, port)

    yield f"http://{host}:{port}/mcp"

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# HTTP transport tests
# ---------------------------------------------------------------------------


class TestHTTPTransport:
    async def test_list_tools(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            expected = {
                "search_docs",
                "search_in_doc",
                "list_docs",
                "get_sections",
                "read_page",
                "read_section",
            }
            assert expected == tool_names

        await _with_session(mcp_http_url, _test)

    async def test_list_prompts(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.list_prompts()
            assert len(result.prompts) == 5

        await _with_session(mcp_http_url, _test)

    async def test_list_resources(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.list_resources()
            uris = {str(r.uri) for r in result.resources}
            assert "docs://status" in uris

        await _with_session(mcp_http_url, _test)

    async def test_call_list_docs(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.call_tool("list_docs", {})
            text = result.content[0].text
            assert "test.pdf" in text

        await _with_session(mcp_http_url, _test)

    async def test_call_search_docs(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.call_tool("search_docs", {"query": "safety"})
            text = result.content[0].text
            assert "Result" in text or "score" in text

        await _with_session(mcp_http_url, _test)

    async def test_call_get_sections(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.call_tool("get_sections", {"filename": "test.pdf"})
            text = result.content[0].text
            assert (
                "test.pdf" in text.lower() or "section" in text.lower() or "1." in text
            )

        await _with_session(mcp_http_url, _test)

    async def test_call_read_page(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.call_tool(
                "read_page", {"filename": "test.pdf", "page": 1}
            )
            text = result.content[0].text
            assert "Page 1" in text or "test.pdf" in text

        await _with_session(mcp_http_url, _test)

    async def test_call_read_section(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.call_tool(
                "read_section",
                {"filename": "test.pdf", "section_title": "Introduction"},
            )
            text = result.content[0].text
            assert len(text) > 0

        await _with_session(mcp_http_url, _test)

    async def test_call_search_in_doc(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.call_tool(
                "search_in_doc", {"query": "specs", "filename": "test.pdf"}
            )
            text = result.content[0].text
            assert len(text) > 0

        await _with_session(mcp_http_url, _test)

    async def test_get_prompt(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.get_prompt(
                "summarize_document", arguments={"filename": "test.pdf"}
            )
            text = result.messages[0].content.text
            assert "get_sections" in text

        await _with_session(mcp_http_url, _test)

    async def test_read_resource_status(self, mcp_http_url: str) -> None:
        async def _test(session: ClientSession) -> None:
            result = await session.read_resource("docs://status")
            text = result.contents[0].text
            assert "Documents: 1" in text

        await _with_session(mcp_http_url, _test)


# ---------------------------------------------------------------------------
# stdio transport tests
# ---------------------------------------------------------------------------


class TestStdioTransport:
    async def test_stdio_list_tools(self, ingested_project: Path) -> None:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        env = {
            **os.environ,
            "PDF2MCP_DATA_DIR": str(ingested_project / "data"),
            "PDF2MCP_DOCS_DIR": str(ingested_project / "docs"),
            "OPENAI_API_KEY": "sk-fake-e2e-key",
        }

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "pdf2mcp", "serve", "--transport", "stdio"],
            env=env,
            cwd=str(ingested_project),
        )

        async with (
            stdio_client(server_params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            assert "list_docs" in tool_names
            assert len(tool_names) == 6

    async def test_stdio_call_list_docs(self, ingested_project: Path) -> None:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        env = {
            **os.environ,
            "PDF2MCP_DATA_DIR": str(ingested_project / "data"),
            "PDF2MCP_DOCS_DIR": str(ingested_project / "docs"),
            "OPENAI_API_KEY": "sk-fake-e2e-key",
        }

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "pdf2mcp", "serve", "--transport", "stdio"],
            env=env,
            cwd=str(ingested_project),
        )

        async with (
            stdio_client(server_params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.call_tool("list_docs", {})
            text = result.content[0].text
            assert "test.pdf" in text
