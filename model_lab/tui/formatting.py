"""Display helpers: every missing value renders explicitly, never as zero."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.markup import escape

from model_lab.application.observability import CostSummary

NA = "N/A"
NOT_MEASURED = "not measured"

STATUS_STYLE = {
    "completed": "green", "passed": "green", "success": "green", "ok": "green", "valid": "green",
    "running": "yellow", "created": "yellow", "partial": "yellow", "incomplete": "yellow", "needs review": "yellow", "draft": "cyan",
    "failed": "red", "cancelled": "red", "provider_failure": "red", "timeout": "red", "infrastructure_failure": "red",
}
PROVENANCE_STYLE = {"SIMULATED": "magenta", "IMPORTED": "cyan", "MEASURED": "green", "PROPOSED": "cyan", "UNKNOWN": "red"}


def text(value: Any, empty: str = NA) -> str:
    """Escape markup; None/empty becomes an explicit placeholder."""
    if value is None or value == "":
        return empty
    return escape(str(value))


def truncate(value: Any, width: int) -> str:
    raw = "" if value is None else str(value).replace("\n", " ")
    return raw if len(raw) <= width else raw[: max(1, width - 1)] + "…"


def pct(value: float | None) -> str:
    return NA if value is None else f"{value * 100:.1f}%"


def ms(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{value:,.1f} ms"


def seconds(value: float | None) -> str:
    if value is None:
        return NA
    if value < 60:
        return f"{value:.1f}s"
    minutes, rest = divmod(int(value), 60)
    return f"{minutes}m{rest:02d}s"


def cost(summary: CostSummary) -> str:
    if summary.total_minor is None:
        return summary.note
    return f"{summary.total_minor / 100:,.2f} {summary.currency} ({summary.note})"


def clock(timestamp: str | None) -> str:
    """Local HH:MM:SS.mmm for an ISO timestamp."""
    if not timestamp:
        return "--:--:--"
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%H:%M:%S.%f")[:-3]
    except ValueError:
        return escape(timestamp)


def styled(value: Any, style_map: dict[str, str] = STATUS_STYLE) -> str:
    label = text(value)
    style = style_map.get(str(value), "")
    return f"[{style}]{label}[/]" if style else label


def bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return escape("[" + "·" * width + "] ") + NA
    filled = round(width * min(done, total) / total)
    return escape(f"[{'█' * filled}{'·' * (width - filled)}]") + f" {done}/{total} {done / total * 100:.0f}%"
