# pdf2mcp

Turn any PDF folder into a searchable MCP server.

## Installation

Clone the repo, then install globally with `uv tool`:

```bash
git clone https://github.com/iSamBa/pdf2mcp.git
uv tool install ./pdf2mcp
```

This makes `pdf2mcp` available as a command anywhere on your system.

To update after pulling new changes:

```bash
uv tool install --force ./pdf2mcp
```

To run directly from source without installing:

```bash
cd ./pdf2mcp
uv run pdf2mcp --help
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

# 4. Get config snippets for your MCP client
pdf2mcp config
```

## Commands

| Command | Description |
|---------|-------------|
| `pdf2mcp init [dir]` | Scaffold a working directory with `docs/` and `.env` |
| `pdf2mcp ingest` | Parse PDFs, chunk, embed, and store in vector DB |
| `pdf2mcp serve` | Start the MCP server (stdio or HTTP) |
| `pdf2mcp config` | Print ready-to-paste config for MCP clients |

### Common Flags

```bash
# Override docs directory
pdf2mcp ingest --docs-dir ./my-pdfs
pdf2mcp serve --docs-dir ./my-pdfs

# HTTP transport
pdf2mcp serve --transport streamable-http --port 9000

# Custom server name
pdf2mcp serve --name my-docs

# Config for a specific client
pdf2mcp config --client cursor --transport streamable-http --port 9000
```

## Client Configuration

`pdf2mcp config` generates ready-to-paste JSON for all supported clients:

| Client | Config File | Top-level Key | HTTP Support |
|--------|------------|--------------|--------------|
| Claude Code | `.mcp.json` | `mcpServers` | Yes |
| Claude Desktop | `claude_desktop_config.json` | `mcpServers` | No (stdio) |
| Cursor | `.cursor/mcp.json` | `mcpServers` | Yes |
| VS Code / Copilot | `.vscode/mcp.json` | `servers` | Yes |

## Environment Variables

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
| `PDF2MCP_SERVER_TRANSPORT` | `stdio` | Transport protocol |
| `PDF2MCP_SERVER_HOST` | `127.0.0.1` | Host for HTTP transport |
| `PDF2MCP_SERVER_PORT` | `8000` | Port for HTTP transport |

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
uv sync --all-extras
uv run pytest
uv run ruff check src/
uv run mypy src/
```
