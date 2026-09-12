#!/usr/bin/env python3
"""Send a test Telegram notification using configured environment variables."""
from methylation_predictor.telegram_notify import (
    format_job_message,
    send_telegram,
    telegram_credentials,
)


def main() -> int:
    token, chat_id = telegram_credentials()
    if not token or not chat_id:
        raise SystemExit(
            "Set METHYLPREDICTOR_TELEGRAM_BOT_TOKEN and "
            "METHYLPREDICTOR_TELEGRAM_CHAT_ID first."
        )
    ok = send_telegram(
        format_job_message(
            "STARTED",
            job_key="telegram/test",
            run_id="telegram-test",
            detail="Telegram integration test",
        ),
        token=token,
        chat_id=chat_id,
        strict=True,
    )
    print("Telegram test sent." if ok else "Telegram test failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
