"""Tests for pdf2mcp.interactive module."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch

import pytest

from pdf2mcp.interactive import (
    WizardCancelledError,
    confirm_prompt,
    print_banner,
    print_step,
    secret_prompt,
    select_prompt,
    text_prompt,
)

# ── WizardCancelledError ─────────────────────────────────────────


class TestWizardCancelledError:
    """Test the WizardCancelledError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(WizardCancelledError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(WizardCancelledError):
            raise WizardCancelledError


# ── text_prompt ──────────────────────────────────────────────────


class TestTextPrompt:
    """Test the text_prompt function."""

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="hello")
    def test_returns_user_input(self, mock_ask: MagicMock) -> None:
        result = text_prompt("Enter value")
        assert result == "hello"

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="mydefault")
    def test_passes_default(self, mock_ask: MagicMock) -> None:
        result = text_prompt("Enter value", default="mydefault")
        assert result == "mydefault"
        mock_ask.assert_called_once_with(
            "Enter value", console=ANY, default="mydefault"
        )

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="typed")
    def test_no_default_omits_default_kwarg(self, mock_ask: MagicMock) -> None:
        result = text_prompt("Enter value")
        assert result == "typed"
        mock_ask.assert_called_once_with("Enter value", console=ANY)

    @patch(
        "pdf2mcp.interactive.Prompt.ask",
        side_effect=KeyboardInterrupt,
    )
    def test_raises_wizard_cancelled_on_ctrl_c(self, mock_ask: MagicMock) -> None:
        with pytest.raises(WizardCancelledError):
            text_prompt("Enter value")


# ── secret_prompt ────────────────────────────────────────────────


class TestSecretPrompt:
    """Test the secret_prompt function."""

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="sk-secret")
    def test_returns_user_input(self, mock_ask: MagicMock) -> None:
        result = secret_prompt("API Key")
        assert result == "sk-secret"

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="sk-secret")
    def test_uses_password_true(self, mock_ask: MagicMock) -> None:
        secret_prompt("API Key")
        mock_ask.assert_called_once_with("API Key", console=ANY, password=True)

    @patch(
        "pdf2mcp.interactive.Prompt.ask",
        side_effect=KeyboardInterrupt,
    )
    def test_raises_wizard_cancelled_on_ctrl_c(self, mock_ask: MagicMock) -> None:
        with pytest.raises(WizardCancelledError):
            secret_prompt("API Key")


# ── confirm_prompt ───────────────────────────────────────────────


class TestConfirmPrompt:
    """Test the confirm_prompt function."""

    @patch("pdf2mcp.interactive.Confirm.ask", return_value=True)
    def test_returns_true(self, mock_ask: MagicMock) -> None:
        assert confirm_prompt("Continue?") is True

    @patch("pdf2mcp.interactive.Confirm.ask", return_value=False)
    def test_returns_false(self, mock_ask: MagicMock) -> None:
        assert confirm_prompt("Continue?") is False

    @patch("pdf2mcp.interactive.Confirm.ask", return_value=True)
    def test_passes_default(self, mock_ask: MagicMock) -> None:
        confirm_prompt("Continue?", default=False)
        mock_ask.assert_called_once_with("Continue?", console=ANY, default=False)

    @patch(
        "pdf2mcp.interactive.Confirm.ask",
        side_effect=KeyboardInterrupt,
    )
    def test_raises_wizard_cancelled_on_ctrl_c(self, mock_ask: MagicMock) -> None:
        with pytest.raises(WizardCancelledError):
            confirm_prompt("Continue?")


# ── select_prompt ────────────────────────────────────────────────


class TestSelectPrompt:
    """Test the select_prompt function."""

    _CHOICES = [
        ("small", "text-embedding-3-small (cheapest)"),
        ("large", "text-embedding-3-large (better quality)"),
        ("ada", "text-embedding-ada-002 (legacy)"),
    ]

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="1")
    def test_returns_first_choice_value(self, mock_ask: MagicMock) -> None:
        result = select_prompt("Model", self._CHOICES)
        assert result == "small"

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="2")
    def test_returns_second_choice_value(self, mock_ask: MagicMock) -> None:
        result = select_prompt("Model", self._CHOICES)
        assert result == "large"

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="3")
    def test_returns_third_choice_value(self, mock_ask: MagicMock) -> None:
        result = select_prompt("Model", self._CHOICES)
        assert result == "ada"

    @patch("pdf2mcp.interactive.Prompt.ask", return_value="1")
    def test_default_sets_prompt_default(self, mock_ask: MagicMock) -> None:
        select_prompt("Model", self._CHOICES, default="large")
        mock_ask.assert_called_once_with(
            "  Select",
            console=ANY,
            default="2",
        )

    @patch("pdf2mcp.interactive.Prompt.ask", side_effect=["abc", "2"])
    def test_invalid_input_reprompts(self, mock_ask: MagicMock) -> None:
        result = select_prompt("Model", self._CHOICES)
        assert result == "large"
        assert mock_ask.call_count == 2

    @patch("pdf2mcp.interactive.Prompt.ask", side_effect=["0", "4", "2"])
    def test_out_of_range_reprompts(self, mock_ask: MagicMock) -> None:
        result = select_prompt("Model", self._CHOICES)
        assert result == "large"
        assert mock_ask.call_count == 3

    @patch(
        "pdf2mcp.interactive.Prompt.ask",
        side_effect=KeyboardInterrupt,
    )
    def test_raises_wizard_cancelled_on_ctrl_c(self, mock_ask: MagicMock) -> None:
        with pytest.raises(WizardCancelledError):
            select_prompt("Model", self._CHOICES)


# ── print_banner ─────────────────────────────────────────────────


class TestPrintBanner:
    """Test the print_banner function."""

    @patch("pdf2mcp.interactive._console")
    def test_prints_without_error(self, mock_console: MagicMock) -> None:
        print_banner()

    @patch("pdf2mcp.interactive._console")
    def test_calls_console_print(self, mock_console: MagicMock) -> None:
        print_banner()
        assert mock_console.print.call_count == 2


# ── print_step ───────────────────────────────────────────────────


class TestPrintStep:
    """Test the print_step function."""

    @patch("pdf2mcp.interactive._console")
    def test_prints_without_error(self, mock_console: MagicMock) -> None:
        print_step(1, 6, "Project Directory")

    @patch("pdf2mcp.interactive._console")
    def test_calls_console_print(self, mock_console: MagicMock) -> None:
        print_step(1, 6, "Project Directory")
        assert mock_console.print.call_count == 3
