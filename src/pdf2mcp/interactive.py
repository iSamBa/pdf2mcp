"""Interactive prompt utilities for the pdf2mcp setup wizard.

Provides reusable prompt functions built on ``rich`` for collecting user input
during interactive CLI flows.  All output is directed to stderr to keep stdout
available for MCP stdio transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

__all__ = [
    "WizardCancelledError",
    "WizardResult",
    "apply_wizard_result",
    "confirm_prompt",
    "generate_env_content",
    "print_banner",
    "print_step",
    "run_wizard",
    "secret_prompt",
    "select_prompt",
    "text_prompt",
]

# All interactive output goes to stderr (stdout reserved for MCP stdio).
_console = Console(stderr=True)

_BANNER = r"""
[bold cyan]██████╗ ██████╗ ███████╗██████╗ ███╗   ███╗ ██████╗██████╗[/]
[bold cyan]██╔══██╗██╔══██╗██╔════╝╚════██╗████╗ ████║██╔════╝██╔══██╗[/]
[bold cyan]██████╔╝██║  ██║█████╗   █████╔╝██╔████╔██║██║     ██████╔╝[/]
[bold cyan]██╔═══╝ ██║  ██║██╔══╝  ██╔═══╝ ██║╚██╔╝██║██║     ██╔═══╝[/]
[bold cyan]██║     ██████╔╝██║     ███████╗██║ ╚═╝ ██║╚██████╗██║[/]
[bold cyan]╚═╝     ╚═════╝ ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝╚═╝[/]
"""


class WizardCancelledError(Exception):
    """Raised when the user cancels the interactive wizard (Ctrl+C)."""


def text_prompt(label: str, default: str | None = None) -> str:
    """Prompt for a text value with an optional default.

    Args:
        label: The prompt label shown to the user.
        default: Default value (shown in brackets, accepted on Enter).

    Returns:
        The user-provided string, or *default* if the user pressed Enter.

    Raises:
        WizardCancelledError: If the user presses Ctrl+C.
    """
    try:
        if default is not None:
            return Prompt.ask(
                label, console=_console, default=default
            )
        return Prompt.ask(label, console=_console)
    except KeyboardInterrupt:
        _console.print()
        raise WizardCancelledError from None


def secret_prompt(label: str) -> str:
    """Prompt for a sensitive value (input masked with ``*``).

    Args:
        label: The prompt label shown to the user.

    Returns:
        The user-provided secret string.

    Raises:
        WizardCancelledError: If the user presses Ctrl+C.
    """
    try:
        return Prompt.ask(label, console=_console, password=True)
    except KeyboardInterrupt:
        _console.print()
        raise WizardCancelledError from None


def confirm_prompt(label: str, default: bool = True) -> bool:
    """Ask a yes/no question.

    Args:
        label: The prompt label shown to the user.
        default: Default answer when the user presses Enter.

    Returns:
        ``True`` for yes, ``False`` for no.

    Raises:
        WizardCancelledError: If the user presses Ctrl+C.
    """
    try:
        return Confirm.ask(label, console=_console, default=default)
    except KeyboardInterrupt:
        _console.print()
        raise WizardCancelledError from None


def select_prompt(
    label: str,
    choices: list[tuple[str, str]],
    default: str | None = None,
) -> str:
    """Present numbered choices and return the selected value.

    Args:
        label: The prompt label shown above the choices.
        choices: List of ``(value, description)`` tuples.
        default: Default value (must match a *value* in *choices*).

    Returns:
        The *value* (first element) of the selected choice.

    Raises:
        WizardCancelledError: If the user presses Ctrl+C.
    """
    _console.print(f"\n  [bold]{label}[/bold]")

    default_number: str | None = None
    for idx, (value, description) in enumerate(choices, 1):
        _console.print(f"    [bold]{idx}[/bold]) {description}")
        if value == default:
            default_number = str(idx)

    err = f"Please enter a number between 1 and {len(choices)}"

    # Keep prompting until we get a valid number.
    while True:
        try:
            if default_number is not None:
                raw: str = Prompt.ask(
                    "  Select",
                    console=_console,
                    default=default_number,
                )
            else:
                raw = Prompt.ask("  Select", console=_console)
        except KeyboardInterrupt:
            _console.print()
            raise WizardCancelledError from None

        try:
            index = int(raw) - 1
        except (ValueError, TypeError):
            _console.print(f"  [red]{err}[/red]")
            continue

        if 0 <= index < len(choices):
            return choices[index][0]

        _console.print(f"  [red]{err}[/red]")


def print_banner() -> None:
    """Print the pdf2mcp ASCII art banner."""
    _console.print(_BANNER)
    _console.print(
        "  [dim]Interactive Setup Wizard[/dim]\n",
    )


def print_step(step: int, total: int, title: str) -> None:
    """Print a numbered step header.

    Displays a rule like ``── [1/6] Project Directory ──``.
    """
    label = Text.from_markup(
        f"[bold green][{step}/{total}][/bold green] {title}"
    )
    _console.print()
    _console.print(Rule(label))
    _console.print()


# ---------------------------------------------------------------------------
# Wizard data model
# ---------------------------------------------------------------------------

_TOTAL_STEPS = 6


@dataclass
class WizardResult:
    """Holds all settings collected by the interactive wizard."""

    target_dir: Path
    openai_api_key: str
    openai_base_url: str
    docs_dir: str
    data_dir: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    server_name: str
    server_transport: str
    server_host: str
    server_port: int
    ocr_enabled: bool
    ocr_language: str
    ocr_dpi: int


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------


def _step_project_dir(target_dir: Path) -> Path:
    """Step 1: Ask for the project directory."""
    print_step(1, _TOTAL_STEPS, "Project Directory")
    result = text_prompt(
        "  Project directory", default=str(target_dir)
    )
    return Path(result)


def _step_openai(
) -> tuple[str, str]:
    """Step 2: Collect OpenAI API key and base URL."""
    print_step(2, _TOTAL_STEPS, "OpenAI API Key")

    while True:
        key = secret_prompt("  OpenAI API key")
        if not key:
            _console.print(
                "  [red]API key cannot be empty[/red]"
            )
            continue
        if not key.startswith("sk-"):
            _console.print(
                "  [red]API key should start with 'sk-'[/red]"
            )
            continue
        break

    base_url = text_prompt(
        "  OpenAI base URL",
        default="https://api.openai.com/v1",
    )
    return key, base_url


def _step_docs_dir() -> str:
    """Step 3: Ask for the documents directory name."""
    print_step(3, _TOTAL_STEPS, "Documents Directory")
    return text_prompt("  Docs directory name", default="docs")


def _step_embedding() -> tuple[str, int, int]:
    """Step 4: Embedding model and chunking settings."""
    print_step(4, _TOTAL_STEPS, "Embedding Settings")

    model = select_prompt(
        "Embedding model",
        [
            (
                "text-embedding-3-small",
                "text-embedding-3-small (cheapest)",
            ),
            (
                "text-embedding-3-large",
                "text-embedding-3-large (better quality)",
            ),
            (
                "text-embedding-ada-002",
                "text-embedding-ada-002 (legacy)",
            ),
        ],
        default="text-embedding-3-small",
    )

    chunk_size_str = text_prompt("  Chunk size", default="500")
    chunk_overlap_str = text_prompt(
        "  Chunk overlap", default="50"
    )

    return model, int(chunk_size_str), int(chunk_overlap_str)


def _step_server() -> tuple[str, str, str, int]:
    """Step 5: Server name, transport, host, and port."""
    print_step(5, _TOTAL_STEPS, "Server Settings")

    name = text_prompt("  Server name", default="pdf-docs")
    transport = select_prompt(
        "Transport protocol",
        [
            (
                "streamable-http",
                "streamable-http (recommended)",
            ),
            ("stdio", "stdio (for Claude Desktop)"),
        ],
        default="streamable-http",
    )

    host = "127.0.0.1"
    port = 8000
    if transport != "stdio":
        host = text_prompt("  Host", default="127.0.0.1")
        port_str = text_prompt("  Port", default="8000")
        port = int(port_str)

    return name, transport, host, port


def _step_ocr() -> tuple[bool, str, int]:
    """Step 6: OCR settings."""
    print_step(6, _TOTAL_STEPS, "OCR Settings")

    enabled = confirm_prompt(
        "  Enable OCR for scanned PDFs?", default=True
    )

    language = "eng"
    dpi = 300
    if enabled:
        language = text_prompt("  OCR language", default="eng")
        dpi_str = text_prompt("  OCR DPI", default="300")
        dpi = int(dpi_str)

    return enabled, language, dpi


# ---------------------------------------------------------------------------
# .env generation
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def generate_env_content(result: WizardResult) -> str:
    """Produce a well-commented ``.env`` file from wizard results.

    The API key is always written. Other settings are written
    uncommented only when they differ from the defaults; defaults
    are shown as commented examples.
    """
    lines: list[str] = []

    def _header(title: str) -> None:
        lines.append(f"# -- {title} " + "-" * (55 - len(title)))
        lines.append("")

    def _setting(
        env_var: str,
        value: object,
        default: object,
        *,
        always: bool = False,
    ) -> None:
        if always or value != default:
            lines.append(f"{env_var}={value}")
        else:
            lines.append(f"# {env_var}={value}")

    _header("OpenAI")
    _setting("OPENAI_API_KEY", result.openai_api_key, "", always=True)
    _setting(
        "PDF2MCP_OPENAI_BASE_URL",
        result.openai_base_url,
        _DEFAULT_BASE_URL,
    )
    lines.append("")

    _header("Paths")
    _setting("PDF2MCP_DOCS_DIR", result.docs_dir, "docs")
    _setting("PDF2MCP_DATA_DIR", result.data_dir, "data")
    lines.append("")

    _header("Embedding & Chunking")
    _setting(
        "PDF2MCP_EMBEDDING_MODEL",
        result.embedding_model,
        "text-embedding-3-small",
    )
    _setting("PDF2MCP_CHUNK_SIZE", result.chunk_size, 500)
    _setting("PDF2MCP_CHUNK_OVERLAP", result.chunk_overlap, 50)
    lines.append("")

    _header("Server")
    _setting("PDF2MCP_SERVER_NAME", result.server_name, "pdf-docs")
    _setting(
        "PDF2MCP_SERVER_TRANSPORT",
        result.server_transport,
        "streamable-http",
    )
    _setting("PDF2MCP_SERVER_HOST", result.server_host, "127.0.0.1")
    _setting("PDF2MCP_SERVER_PORT", result.server_port, 8000)
    lines.append("")

    _header("OCR")
    _setting(
        "PDF2MCP_OCR_ENABLED",
        str(result.ocr_enabled).lower(),
        "true",
    )
    _setting("PDF2MCP_OCR_LANGUAGE", result.ocr_language, "eng")
    _setting("PDF2MCP_OCR_DPI", result.ocr_dpi, 300)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def _print_summary(result: WizardResult) -> None:
    """Display a summary table of all collected settings."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    masked_key = result.openai_api_key[:7] + "..." + result.openai_api_key[-4:]
    table.add_row("Project directory", str(result.target_dir))
    table.add_row("OpenAI API key", masked_key)
    table.add_row("OpenAI base URL", result.openai_base_url)
    table.add_row("Docs directory", result.docs_dir)
    table.add_row("Embedding model", result.embedding_model)
    table.add_row(
        "Chunk size / overlap",
        f"{result.chunk_size} / {result.chunk_overlap}",
    )
    table.add_row("Server name", result.server_name)
    table.add_row("Transport", result.server_transport)
    if result.server_transport != "stdio":
        table.add_row(
            "Host / Port",
            f"{result.server_host}:{result.server_port}",
        )
    ocr_status = "enabled" if result.ocr_enabled else "disabled"
    table.add_row("OCR", ocr_status)
    if result.ocr_enabled:
        table.add_row(
            "OCR language / DPI",
            f"{result.ocr_language} / {result.ocr_dpi}",
        )

    _console.print()
    _console.print(
        Panel(table, title="Configuration Summary", border_style="green")
    )
    _console.print()


# ---------------------------------------------------------------------------
# Main wizard flow
# ---------------------------------------------------------------------------


def run_wizard(target_dir: Path) -> WizardResult:
    """Run the interactive setup wizard.

    Walks the user through 6 steps to collect all pdf2mcp settings.

    Args:
        target_dir: Initial target directory (can be overridden in step 1).

    Returns:
        A :class:`WizardResult` with all collected settings.

    Raises:
        WizardCancelledError: If the user presses Ctrl+C at any step.
    """
    print_banner()

    target = _step_project_dir(target_dir)
    api_key, base_url = _step_openai()
    docs_dir = _step_docs_dir()
    model, chunk_size, chunk_overlap = _step_embedding()
    name, transport, host, port = _step_server()
    ocr_enabled, ocr_lang, ocr_dpi = _step_ocr()

    return WizardResult(
        target_dir=target,
        openai_api_key=api_key,
        openai_base_url=base_url,
        docs_dir=docs_dir,
        data_dir="data",
        embedding_model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        server_name=name,
        server_transport=transport,
        server_host=host,
        server_port=port,
        ocr_enabled=ocr_enabled,
        ocr_language=ocr_lang,
        ocr_dpi=ocr_dpi,
    )


def apply_wizard_result(result: WizardResult) -> None:
    """Create directories and write ``.env`` from wizard results.

    Displays a summary panel, asks for confirmation, then writes
    the project scaffold.

    Raises:
        WizardCancelledError: If the user declines confirmation.
    """
    _print_summary(result)

    if not confirm_prompt("  Write these settings?", default=True):
        raise WizardCancelledError

    target = result.target_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / result.docs_dir).mkdir(parents=True, exist_ok=True)

    env_path = target / ".env"
    if env_path.exists():
        if not confirm_prompt(
            f"  {env_path} already exists. Overwrite?",
            default=False,
        ):
            _console.print("  [dim]Skipped .env (kept existing)[/dim]")
        else:
            env_path.write_text(generate_env_content(result))
            _console.print(f"  [green]Wrote {env_path}[/green]")
    else:
        env_path.write_text(generate_env_content(result))
        _console.print(f"  [green]Wrote {env_path}[/green]")

    _console.print(
        f"  [green]Created {target / result.docs_dir}/[/green]"
    )
    _console.print()
    _console.print(
        Panel(
            f"[bold green]Setup complete![/bold green]\n\n"
            f"Next steps:\n"
            f"  1. Add your PDFs to [bold]{target / result.docs_dir}/[/bold]\n"
            f"  2. Run [bold]pdf2mcp ingest[/bold]\n"
            f"  3. Run [bold]pdf2mcp serve[/bold]",
            border_style="green",
        )
    )
