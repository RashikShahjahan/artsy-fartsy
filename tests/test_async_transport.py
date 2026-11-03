"""Deadlines and cancellation actually release the HTTP response stream."""
import asyncio
import unittest

import httpx

from drawing.dataset_client import ChatClient, close_chat_clients


class WaitingBody(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    async def aclose(self):
        self.closed = True


class AsyncTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_and_deadline_close_response(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                body = WaitingBody()
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda req: httpx.Response(200, stream=body))) as http:
                    client = ChatClient("teacher", "https://example.invalid/v1", http_client=http,
                                        timeout_ms=20 if timeout else 60_000)
                    task = asyncio.create_task(client.complete([], 10))
                    await body.started.wait()
                    if timeout:
                        with self.assertRaisesRegex(RuntimeError, "timed out"):
                            await task
                    else:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    self.assertTrue(body.closed)

    async def test_transport_errors_do_not_expose_provider_text(self):
        def fail(req):
            raise httpx.ConnectError("provider echoed a-secret", request=req)
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            client = ChatClient("teacher", "https://example.invalid/v1", http_client=http)
            with self.assertRaisesRegex(RuntimeError, "transport failed") as error:
                await client.complete([], 10)
            self.assertNotIn("a-secret", str(error.exception))

    async def test_run_cleanup_closes_owned_pool(self):
        from unittest.mock import patch
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]})))
        with patch("drawing.dataset_client.httpx.AsyncClient", return_value=http) as factory:
            client = ChatClient("teacher", "https://example.invalid/v1")
            await client.complete([], 10)
            await client.complete([], 10)
            factory.assert_called_once()
            await close_chat_clients()
            self.assertTrue(http.is_closed)
            self.assertIsNone(client.http_client)
