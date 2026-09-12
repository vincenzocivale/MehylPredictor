import urllib.parse

from methylation_predictor.telegram_notify import (
    format_job_message,
    send_telegram,
)


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return b'{"ok": true}'


def test_telegram_post():
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response()

    assert send_telegram(
        "hello",
        token="SECRET",
        chat_id="123",
        opener=opener,
    )
    request, _ = calls[0]
    body = urllib.parse.parse_qs(request.data.decode())
    assert body["chat_id"] == ["123"]
    assert body["text"] == ["hello"]


def test_message_contains_metrics():
    text = format_job_message(
        "COMPLETED",
        job_key="main/main/seed17",
        run_id="main__main__seed17",
        machine="hal",
        gpu="0",
        headline_metrics={"mas_pcc": 0.6},
    )
    assert "main/main/seed17" in text
    assert "mas_pcc: 0.600000" in text
