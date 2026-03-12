# pdf2mcp

```
██████╗ ██████╗ ███████╗██████╗ ███╗   ███╗ ██████╗██████╗
██╔══██╗██╔══██╗██╔════╝╚════██╗████╗ ████║██╔════╝██╔══██╗
██████╔╝██║  ██║█████╗   █████╔╝██╔████╔██║██║     ██████╔╝
██╔═══╝ ██║  ██║██╔══╝  ██╔═══╝ ██║╚██╔╝██║██║     ██╔═══╝
██║     ██████╔╝██║     ███████╗██║ ╚═╝ ██║╚██████╗██║
╚═╝     ╚═════╝ ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝╚═╝
```

[![PyPI](https://img.shields.io/pypi/v/pdf2mcp)](https://pypi.org/project/pdf2mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Turn any PDF folder into a searchable MCP server with semantic search.

## Installation

### From PyPI (recommended)

```bash
pip install pdf2mcp
```

Or with `uv`:

```bash
uv tool install pdf2mcp
```

### From source

```bash
git clone https://github.com/iSamBa/pdf2mcp.git
uv tool install ./pdf2mcp
```

To update after pulling new changes:

```bash
uv tool install --force ./pdf2mcp
```

### Verify

```bash
pdf2mcp --version
```

## Quick Start

```bash
# 1. Scaffold a project (creates docs/ and .env)
pdf2mcp init ./my-project
cd my-project

# 2. Add your PDFs to docs/ and set OPENAI_API_KEY in .env

# 3. Ingest
pdf2mcp ingest

# 4. Start the server
pdf2mcp serve

# 5. Get config snippets for your MCP client
pdf2mcp config
```

## Architecture

pdf2mcp separates **server** and **client** concerns:

- **Server** (`pdf2mcp serve`) — runs independently, handles PDF ingestion, embedding, and search. Configured via `PDF2MCP_*` environment variables.
- **Client** (Claude Code, Cursor, VS Code, etc.) — connects to a running server over HTTP. Only needs the server URL.

The default transport is `streamable-http`. The server listens on `http://127.0.0.1:8000/mcp` and shuts down gracefully on SIGINT/SIGTERM.

## Commands

| Command | Description |
|---------|-------------|
| `pdf2mcp init [dir]` | Scaffold a working directory with `docs/` and `.env` |
| `pdf2mcp ingest` | Parse PDFs, chunk, embed, and store in vector DB |
| `pdf2mcp serve` | Start the MCP server (HTTP by default) |
| `pdf2mcp config` | Print ready-to-paste config for MCP clients |

### Common Flags

```bash
# Override docs directory
pdf2mcp ingest --docs-dir ./my-pdfs
pdf2mcp serve --docs-dir ./my-pdfs

# Use stdio transport (for clients that spawn the server)
pdf2mcp serve --transport stdio

# Custom host/port
pdf2mcp serve --host 0.0.0.0 --port 9000

# Custom server name
pdf2mcp serve --name my-docs

# Config for a specific client
pdf2mcp config --client cursor
pdf2mcp config --client claude-desktop --transport stdio
```

## Client Configuration

`pdf2mcp config` generates ready-to-paste JSON for all supported clients. The default is HTTP — clients just need the server URL:

```json
{
  "mcpServers": {
    "pdf-docs": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

| Client | Config File | Top-level Key | HTTP Support |
|--------|------------|--------------|--------------|
| Claude Code | `.mcp.json` | `mcpServers` | Yes |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | No (stdio only) |
| Cursor | `.cursor/mcp.json` | `mcpServers` | Yes |
| VS Code / Copilot | `.vscode/mcp.json` | `servers` | Yes |

Use `--transport stdio` for clients that need to spawn the server process (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "pdf-docs": {
      "command": "uv",
      "args": ["run", "pdf2mcp", "serve"]
    }
  }
}
```

## Environment Variables

### Server settings (`PDF2MCP_*`)

These configure the server process. MCP clients never need these.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings |
| `PDF2MCP_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL (for Azure, local proxies, or compatible providers) |
| `PDF2MCP_DOCS_DIR` | `docs` | Directory containing PDF files |
| `PDF2MCP_DATA_DIR` | `data` | Directory for vector database |
| `PDF2MCP_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `PDF2MCP_CHUNK_SIZE` | `500` | Target chunk size in tokens |
| `PDF2MCP_CHUNK_OVERLAP` | `50` | Overlap between chunks in tokens |
| `PDF2MCP_DEFAULT_NUM_RESULTS` | `5` | Default search results count |
| `PDF2MCP_SERVER_NAME` | `pdf-docs` | MCP server name |
| `PDF2MCP_SERVER_TRANSPORT` | `streamable-http` | Transport protocol |
| `PDF2MCP_SERVER_HOST` | `127.0.0.1` | Host to bind to |
| `PDF2MCP_SERVER_PORT` | `8000` | Port to bind to |

### Client settings (`PDF2MCP_CLIENT_*`)

These configure how a client connects to the server. No secrets needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF2MCP_CLIENT_SERVER_NAME` | `pdf-docs` | Server name in client config |
| `PDF2MCP_CLIENT_SERVER_URL` | `http://127.0.0.1:8000/mcp` | Server URL |
| `PDF2MCP_CLIENT_TRANSPORT` | `streamable-http` | Transport protocol |

## MCP Tools

The server exposes six tools:

| Tool | Description |
|------|-------------|
| `search_docs(query)` | Semantic search across **all** ingested PDFs |
| `search_in_doc(query, filename)` | Semantic search scoped to a **single** document |
| `list_docs()` | List all ingested documents with chunk counts |
| `get_sections(filename)` | Get section headings for a specific document |
| `read_page(filename, page)` | Read the full content of a specific page |
| `read_section(filename, section_title)` | Read the full content of a named section |

### Typical workflow

1. **`list_docs`** — discover available documents
2. **`get_sections`** — browse a document's structure
3. **`read_section`** or **`read_page`** — read specific content
4. **`search_docs`** or **`search_in_doc`** — find information by query

## Development

```bash
git clone https://github.com/iSamBa/pdf2mcp.git
cd pdf2mcp
uv sync --all-extras
uv run pytest
uv run ruff check src/
uv run mypy src/
```

## License

[MIT](LICENSE)
