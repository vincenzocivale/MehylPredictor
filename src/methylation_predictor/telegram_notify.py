"""Best-effort Telegram notifications for distributed paper jobs."""
from __future__ import annotations

import json
import os
import socket
import urllib.parse
import urllib.request
from typing import Any, Callable


TOKEN_ENV = (
    "METHYLPREDICTOR_TELEGRAM_BOT_TOKEN",
    "TELEGRAM_BOT_TOKEN",
)
CHAT_ENV = (
    "METHYLPREDICTOR_TELEGRAM_CHAT_ID",
    "TELEGRAM_CHAT_ID",
)


def first_env(names: tuple[str, ...] | list[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def telegram_credentials() -> tuple[str | None, str | None]:
    return first_env(TOKEN_ENV), first_env(CHAT_ENV)


def send_telegram(
    message: str,
    *,
    token: str | None = None,
    chat_id: str | None = None,
    timeout: float = 10.0,
    strict: bool = False,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    token = token or first_env(TOKEN_ENV)
    chat_id = chat_id or first_env(CHAT_ENV)
    if not token or not chat_id:
        return False

    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode(
            {
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            raw = response.read()
        if raw:
            payload = json.loads(raw.decode("utf-8"))
            if not payload.get("ok", False):
                raise RuntimeError(f"Telegram ok=false: {payload}")
        return True
    except Exception as exc:
        if strict:
            raise
        print(f"[telegram] warning: {exc}", flush=True)
        return False


def format_job_message(
    event: str,
    *,
    job_key: str,
    run_id: str,
    machine: str | None = None,
    gpu: str | None = None,
    duration_seconds: float | None = None,
    headline_metrics: dict[str, Any] | None = None,
    detail: str | None = None,
) -> str:
    icon = {
        "STARTED": "🚀",
        "COMPLETED": "✅",
        "ERROR": "❌",
        "PUSH_ERROR": "⚠️",
        "RECOVERED": "♻️",
    }.get(event, "ℹ️")
    lines = [
        f"{icon} MethylPredictor — {event}",
        f"job: {job_key}",
        f"run: {run_id}",
        f"host: {machine or socket.gethostname()}",
    ]
    if gpu is not None:
        lines.append(f"gpu: {gpu}")
    if duration_seconds is not None:
        lines.append(f"duration: {duration_seconds / 3600.0:.2f} h")
    if headline_metrics:
        for key in ("mas_pcc", "mac_pcc", "mse", "mae", "skill_vs_prior"):
            value = headline_metrics.get(key)
            if value is not None:
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.6f}")
                else:
                    lines.append(f"{key}: {value}")
    if detail:
        lines.append(f"detail: {detail[:1200]}")
    return "\n".join(lines)
