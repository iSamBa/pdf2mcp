"""Interactive prompt utilities for the pdf2mcp setup wizard.

Provides reusable prompt functions built on ``rich`` for collecting user input
during interactive CLI flows.  All output is directed to stderr to keep stdout
available for MCP stdio transport.
"""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.text import Text

__all__ = [
    "WizardCancelledError",
    "confirm_prompt",
    "print_banner",
    "print_step",
    "secret_prompt",
    "select_prompt",
    "text_prompt",
]

# All interactive output goes to stderr (stdout reserved for MCP stdio).
console = Console(stderr=True)

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
                label, console=console, default=default
            )
        return Prompt.ask(label, console=console)
    except KeyboardInterrupt:
        console.print()
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
        return Prompt.ask(label, console=console, password=True)
    except KeyboardInterrupt:
        console.print()
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
        return Confirm.ask(label, console=console, default=default)
    except KeyboardInterrupt:
        console.print()
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
    console.print(f"\n  [bold]{label}[/bold]")

    default_number: str | None = None
    for idx, (value, description) in enumerate(choices, 1):
        console.print(f"    [bold]{idx}[/bold]) {description}")
        if value == default:
            default_number = str(idx)

    err = f"Please enter a number between 1 and {len(choices)}"

    # Keep prompting until we get a valid number.
    while True:
        try:
            if default_number is not None:
                raw: str = Prompt.ask(
                    "  Select",
                    console=console,
                    default=default_number,
                )
            else:
                raw = Prompt.ask("  Select", console=console)
        except KeyboardInterrupt:
            console.print()
            raise WizardCancelledError from None

        try:
            index = int(raw) - 1
        except (ValueError, TypeError):
            console.print(f"  [red]{err}[/red]")
            continue

        if 0 <= index < len(choices):
            return choices[index][0]

        console.print(f"  [red]{err}[/red]")


def print_banner() -> None:
    """Print the pdf2mcp ASCII art banner."""
    console.print(_BANNER)
    console.print(
        "  [dim]Interactive Setup Wizard[/dim]\n",
    )


def print_step(step: int, total: int, title: str) -> None:
    """Print a numbered step header.

    Displays a rule like ``── [1/6] Project Directory ──``.
    """
    label = Text.from_markup(
        f"[bold green][{step}/{total}][/bold green] {title}"
    )
    console.print()
    console.print(Rule(label))
    console.print()
