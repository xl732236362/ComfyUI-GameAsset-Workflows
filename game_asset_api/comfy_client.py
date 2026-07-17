"""Typed asynchronous client for the local ComfyUI prompt API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from urllib.parse import quote

import aiohttp


ImageRecord = dict[str, object]


class ComfyClient:
    """Submit prompt graphs and wait for their completed ComfyUI history entries."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        session: aiohttp.ClientSession | None = None,
        poll_interval_seconds: float = 2,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._sleep = sleep or asyncio.sleep

    async def close(self) -> None:
        """Close the session when this client created it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> ComfyClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def submit(self, graph: Mapping[str, object]) -> str:
        """Submit *graph* and return ComfyUI's non-empty prompt identifier."""
        try:
            session = self._get_session()
            async with session.post(f"{self.base_url}/prompt", json={"prompt": graph}) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"ComfyUI prompt request failed with status {response.status}"
                    )
                payload = await _json_object(response, "prompt response")
        except aiohttp.ClientError as error:
            raise RuntimeError("ComfyUI prompt request failed") from error

        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise RuntimeError("ComfyUI returned a malformed prompt response")
        return prompt_id

    async def wait_for_prompt(
        self, prompt_id: str, timeout_seconds: float = 1800
    ) -> dict[str, object]:
        """Poll ComfyUI history until *prompt_id* completes or the deadline expires."""
        if timeout_seconds <= 0:
            raise TimeoutError("ComfyUI prompt timed out")

        clock = self._clock or asyncio.get_running_loop().time
        deadline = clock() + timeout_seconds
        history_url = f"{self.base_url}/history/{quote(prompt_id, safe='')}"
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("ComfyUI prompt timed out")
            try:
                session = self._get_session()
                request_timeout = aiohttp.ClientTimeout(total=remaining)
                async with session.get(history_url, timeout=request_timeout) as response:
                    if not 200 <= response.status < 300:
                        raise RuntimeError(
                            f"ComfyUI history request failed with status {response.status}"
                        )
                    payload = await _json_object(response, "history response")
            except TimeoutError as error:
                raise TimeoutError("ComfyUI prompt timed out") from error
            except aiohttp.ClientError as error:
                raise RuntimeError("ComfyUI history request failed") from error

            entry = _history_entry(payload, prompt_id)
            if entry is not None:
                error_message = _execution_error(entry)
                if error_message is not None:
                    raise RuntimeError(error_message)
                if "outputs" in entry:
                    return entry

            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("ComfyUI prompt timed out")
            await self._sleep(min(self._poll_interval_seconds, remaining))

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session


async def _json_object(
    response: aiohttp.ClientResponse, response_name: str
) -> dict[str, object]:
    try:
        payload = await response.json(content_type=None)
    except (aiohttp.ClientError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"ComfyUI returned a malformed {response_name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"ComfyUI returned a malformed {response_name}")
    return payload


def _history_entry(
    payload: Mapping[str, object], prompt_id: str
) -> dict[str, object] | None:
    candidate = payload.get(prompt_id)
    if candidate is None and "outputs" in payload:
        candidate = payload
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise RuntimeError("ComfyUI returned a malformed history response")
    return dict(candidate)


def _execution_error(history: Mapping[str, object]) -> str | None:
    status = history.get("status")
    if not isinstance(status, Mapping):
        return None
    messages = status.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
            detail = message[1] if len(message) > 1 else None
            return f"ComfyUI execution error: {_sanitize_error(detail)}"
    return None


def _sanitize_error(detail: object) -> str:
    if isinstance(detail, Mapping):
        for key in ("exception_message", "message", "error"):
            value = detail.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:500]
    if isinstance(detail, str) and detail.strip():
        return " ".join(detail.split())[:500]
    return "execution failed"


def image_records(history: Mapping[str, object], node_id: str) -> list[ImageRecord]:
    """Return validated image records emitted by one completed workflow node."""
    outputs = history.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("history outputs are malformed")
    node = outputs.get(node_id)
    if not isinstance(node, Mapping):
        raise ValueError(f"history output node {node_id} is missing")
    images = node.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"history output node {node_id} images are missing or malformed")
    if any(not isinstance(image, Mapping) for image in images):
        raise ValueError(f"history output node {node_id} images are malformed")
    return [dict(image) for image in images]
