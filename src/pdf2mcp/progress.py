"""Progress reporting for the ingestion pipeline."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

__all__ = ["IngestionProgress"]

_STAGE_STYLES = {
    "parsing": "cyan",
    "chunking": "yellow",
    "embedding": "magenta",
    "storing": "green",
    "done": "bold green",
    "skipped": "dim",
}


class _StageColumn(TextColumn):
    """Displays the current processing stage with color coding."""

    def __init__(self) -> None:
        super().__init__("")

    def render(self, task: object) -> Text:  # type: ignore[override]
        stage: str = task.fields.get("stage", "")  # type: ignore[union-attr]
        if not stage:
            return Text("")
        style = _STAGE_STYLES.get(stage, "white")
        return Text(f"[{stage}]", style=style)


class IngestionProgress:
    """Rich-based progress display for the ingestion pipeline.

    Per-document progress is weighted by actual work units:
    1 (parse) + 1 (chunk) + N (embedding batches) + 1 (store).
    This gives smooth, granular progress during the embedding phase
    which is typically the bottleneck.

    Usage::

        with IngestionProgress(total_docs=len(pdfs)) as progress:
            for pdf in pdfs:
                progress.document_start(pdf.name)
                progress.stage_start("parsing")
                ...
                progress.stage_complete()
                # After chunking, set the real embedding batch count
                progress.set_embedding_batches(num_batches)
                progress.stage_start("embedding")
                # Call advance_embedding() after each batch
                progress.advance_embedding()
                ...
                progress.document_complete()
    """

    # Base steps: parse(1) + chunk(1) + store(1) = 3.
    # Embedding steps are added dynamically after chunking.
    _BASE_STEPS = 3

    def __init__(self, total_docs: int) -> None:
        self._total_docs = total_docs
        self._console = Console(stderr=True)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            _StageColumn(),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._overall_task = None
        self._doc_task = None
        self._saved_handlers: list[logging.Handler] = []

    def __enter__(self) -> IngestionProgress:
        # Replace all logging handlers with a Rich handler that renders
        # log messages above the progress bars without visual corruption.
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        root.handlers.clear()
        rich_handler = RichHandler(
            console=self._console,
            show_path=False,
            show_time=False,
            markup=False,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(logging.WARNING)
        root.addHandler(rich_handler)

        self._progress.__enter__()
        self._overall_task = self._progress.add_task(
            "Ingesting documents", total=self._total_docs, stage=""
        )
        return self

    def __exit__(self, *args: object) -> None:
        self._progress.__exit__(*args)  # type: ignore[arg-type]

        # Restore original logging handlers.
        root = logging.getLogger()
        root.handlers.clear()
        for handler in self._saved_handlers:
            root.addHandler(handler)
        self._saved_handlers.clear()

    def document_start(self, filename: str) -> None:
        """Signal start of processing a new document."""
        # Start with base steps; total is adjusted once we know the
        # number of embedding batches.
        self._doc_task = self._progress.add_task(
            f"  {filename}", total=self._BASE_STEPS, stage=""
        )

    def stage_start(self, stage: str) -> None:
        """Signal start of a processing stage for the current document."""
        if self._doc_task is not None:
            self._progress.update(self._doc_task, stage=stage)

    def stage_complete(self) -> None:
        """Signal completion of the current stage (parse, chunk, or store)."""
        if self._doc_task is not None:
            self._progress.advance(self._doc_task)

    def set_embedding_batches(self, num_batches: int) -> None:
        """Update the document total to reflect the real embedding work.

        Call this after chunking, once the number of embedding batches
        is known.  The total becomes: base_steps + num_batches.
        """
        if self._doc_task is not None:
            self._progress.update(
                self._doc_task, total=self._BASE_STEPS + num_batches
            )

    def advance_embedding(self) -> None:
        """Advance progress by one embedding batch."""
        if self._doc_task is not None:
            self._progress.advance(self._doc_task)

    def document_complete(self) -> None:
        """Signal completion of the current document."""
        if self._doc_task is not None:
            self._progress.update(self._doc_task, stage="done")
            self._doc_task = None
        if self._overall_task is not None:
            self._progress.advance(self._overall_task)

    def document_skipped(self, filename: str) -> None:
        """Signal that a document was skipped (unchanged)."""
        if self._doc_task is not None:
            self._progress.update(
                self._doc_task, stage="skipped", completed=self._BASE_STEPS
            )
            self._doc_task = None
        if self._overall_task is not None:
            self._progress.advance(self._overall_task)
