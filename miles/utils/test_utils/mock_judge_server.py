"""Mock OpenAI-compatible judge LLM server for testing llm_judge RM.

Provides a configurable /v1/chat/completions endpoint that returns
judge verdicts based on a pluggable judge_fn.
"""

from collections.abc import Callable
from contextlib import contextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

JudgeFn = Callable[[str], str]


def default_judge_fn(user_message: str) -> str:
    """Default mock judge: always returns tie."""
    return "A=B"


class MockJudgeServer:
    def __init__(
        self,
        judge_fn: JudgeFn = default_judge_fn,
        host: str = "127.0.0.1",
        port: int | None = None,
    ):
        self.judge_fn = judge_fn
        self.host = host
        self.port = port or find_available_port(31000)
        self.request_log: list[dict] = []
        self.header_log: list[dict] = []

        self.app = FastAPI()
        self._server: UvicornThreadServer | None = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            payload = await request.json()
            self.request_log.append(payload)
            self.header_log.append(dict(request.headers))

            messages = payload.get("messages", [])
            user_message = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")

            verdict_text = self.judge_fn(user_message)

            response = {
                "id": "mock-judge-001",
                "object": "chat.completion",
                "model": payload.get("model", "mock-judge"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": verdict_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            return JSONResponse(content=response)

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.url}/v1/chat/completions"

    def start(self):
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._server.start()

    def stop(self):
        if self._server is not None:
            self._server.stop()

    def reset_stats(self):
        self.request_log.clear()
        self.header_log.clear()


@contextmanager
def with_mock_judge_server(
    judge_fn: JudgeFn = default_judge_fn,
    host: str = "127.0.0.1",
    port: int | None = None,
):
    server = MockJudgeServer(judge_fn=judge_fn, host=host, port=port)
    try:
        server.start()
        yield server
    finally:
        server.stop()
