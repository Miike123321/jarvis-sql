"""MCP client for connecting Jarvis to an Apache Superset MCP server."""

import asyncio
import concurrent.futures
import json
import logging
import threading
from collections.abc import Awaitable, Collection
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from config_manager import SupersetMCPConfig

logger = logging.getLogger("jarvis.superset_mcp")

CONNECT_TIMEOUT_SECONDS = 10
TOOL_TIMEOUT_SECONDS = 30
CLOSE_TIMEOUT_SECONDS = 5
MAX_SUPERSET_TOOL_RESULT_CHARS = 12000

SUPPORTED_SUPERSET_TOOLS = (
    "health_check",
    "get_instance_info",
    "get_schema",
    "list_databases",
    "get_database_info",
    "list_datasets",
    "get_dataset_info",
    "query_dataset",
    "create_virtual_dataset",
    "list_charts",
    "get_chart_info",
    "get_chart_data",
    "get_chart_preview",
    "get_chart_sql",
    "get_chart_type_schema",
    "generate_explore_link",
    "generate_chart",
    "update_chart",
    "update_chart_preview",
    "list_dashboards",
    "get_dashboard_info",
    "get_dashboard_data",
    "generate_dashboard",
    "add_chart_to_existing_dashboard",
    "update_dashboard",
    "execute_sql",
    "open_sql_lab_with_context",
    "save_sql_query",
)

SUPERSET_REQUESTED_PERMISSIONS = (
    "can_read on Chart",
    "can_write on Chart",
    "can_read on Dashboard",
    "can_write on Dashboard",
    "can_read on Dataset",
    "can_write on Dataset",
    "can_read on Database",
    "can_read on SQLLab",
    "can_execute_sql_query on SQLLab",
    "can_estimate_query_cost on SQLLab",
    "can_format_sql on SQLLab",
    "can_write on SavedQuery",
    "database_access",
    "datasource_access",
)

MUTATING_SUPERSET_TOOLS = frozenset(
    {
        "create_virtual_dataset",
        "generate_chart",
        "update_chart",
        "update_chart_preview",
        "generate_dashboard",
        "add_chart_to_existing_dashboard",
        "update_dashboard",
        "save_sql_query",
    }
)


class SupersetMCPError(RuntimeError):
    """A safe, user-facing MCP connection or request failure."""


def validate_superset_config(config: SupersetMCPConfig) -> None:
    """Validate a user-supplied Streamable HTTP endpoint and credentials."""
    if not config.enabled:
        raise SupersetMCPError("Интеграция Superset MCP отключена.")
    if not config.endpoint_url:
        raise SupersetMCPError(
            "Укажите точный Streamable HTTP endpoint MCP-сервера Superset."
        )

    try:
        parsed = urlsplit(config.endpoint_url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise SupersetMCPError("Некорректный URL MCP endpoint.") from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise SupersetMCPError(
            "Endpoint должен быть HTTP(S)-URL без логина, пароля, query-параметров "
            "и fragment."
        )
    if parsed.scheme == "http" and hostname.lower() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise SupersetMCPError(
            "Для удалённого MCP endpoint требуется HTTPS; HTTP разрешён только "
            "для localhost."
        )


def _connection_error_message(exc: Exception) -> str:
    nested_errors = getattr(exc, "exceptions", ())
    for nested_error in nested_errors:
        if isinstance(nested_error, Exception):
            message = _connection_error_message(nested_error)
            if message != (
                "Не удалось установить MCP-соединение с Superset. Проверьте endpoint, "
                "аутентификацию и настройки сервера."
            ):
                return message
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in {401, 403}:
            return (
                f"Superset отклонил аутентификацию или доступ (HTTP {status}). "
                "Проверьте Bearer-токен и права пользователя."
            )
        return f"MCP endpoint вернул HTTP {status}. Проверьте endpoint и настройки сервера."
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "Истекло время ожидания ответа MCP-сервера Superset."
    if isinstance(exc, httpx.ConnectError):
        return "Не удалось подключиться к MCP endpoint Superset. Проверьте URL и сеть."
    return (
        "Не удалось установить MCP-соединение с Superset. Проверьте endpoint, "
        "аутентификацию и настройки сервера."
    )


class SupersetMCPClient:
    """A reusable Streamable HTTP MCP session hosted on a dedicated event loop."""

    def __init__(
        self,
        config: SupersetMCPConfig,
        *,
        httpx_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ):
        validate_superset_config(config)
        self._endpoint_url = config.endpoint_url
        self._bearer_token = config.bearer_token.strip()
        self._httpx_client_factory = httpx_client_factory
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="superset-mcp-client",
            daemon=True,
        )
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._shutdown_event: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._startup_error: Exception | None = None
        self._runtime_error: Exception | None = None
        self._thread.start()

        if not self._ready.wait(CONNECT_TIMEOUT_SECONDS + 1):
            self.close()
            raise SupersetMCPError(
                "Истекло время ожидания инициализации MCP-сервера Superset."
            )
        if self._startup_error:
            error = self._startup_error
            self.close()
            raise SupersetMCPError(_connection_error_message(error)) from error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            if self._ready.is_set():
                self._runtime_error = exc
            else:
                self._startup_error = exc
            self._ready.set()
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception as exc:
                logger.warning(
                    "Superset MCP event-loop cleanup failed (%s)",
                    type(exc).__name__,
                )
            self._loop.close()
            self._closed.set()

    async def _serve(self) -> None:
        self._shutdown_event = asyncio.Event()
        headers = (
            {"Authorization": f"Bearer {self._bearer_token}"}
            if self._bearer_token
            else None
        )
        timeout = httpx.Timeout(
            TOOL_TIMEOUT_SECONDS,
            connect=CONNECT_TIMEOUT_SECONDS,
            pool=CONNECT_TIMEOUT_SECONDS,
        )
        http_client = self._httpx_client_factory(
            headers=headers,
            timeout=timeout,
        )

        async with http_client:
            async with streamable_http_client(
                self._endpoint_url,
                http_client=http_client,
                terminate_on_close=True,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(
                        session.initialize(),
                        timeout=CONNECT_TIMEOUT_SECONDS,
                    )
                    self._session = session
                    self._ready.set()
                    await self._shutdown_event.wait()

    async def _list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            raise SupersetMCPError("MCP-сессия Superset закрыта.")
        response = await self._session.list_tools()
        return [
            tool.model_dump(by_alias=True, mode="json")
            for tool in response.tools
        ]

    async def _call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Any:
        if self._session is None:
            raise SupersetMCPError("MCP-сессия Superset закрыта.")
        return await self._session.call_tool(name, arguments)

    def _submit(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        if self._runtime_error:
            raise SupersetMCPError(
                _connection_error_message(self._runtime_error)
            ) from self._runtime_error
        if self._closed.is_set() or not self._thread.is_alive():
            raise SupersetMCPError("MCP-сессия Superset закрыта.")

        coroutine = operation()
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=TOOL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise SupersetMCPError(
                "Истекло время ожидания ответа инструмента Superset MCP."
            ) from exc
        except SupersetMCPError:
            raise
        except Exception as exc:
            raise SupersetMCPError(_connection_error_message(exc)) from exc

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's current MCP tool metadata."""
        return self._submit(self._list_tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the existing MCP session."""
        return self._submit(lambda: self._call_tool(name, arguments))

    async def _request_shutdown(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    def close(self) -> None:
        """Stop the session and close its HTTP client."""
        if not self._thread.is_alive():
            return
        if self._shutdown_event is not None and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._request_shutdown(), self._loop
            )
            try:
                future.result(timeout=CLOSE_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning("Timed out requesting Superset MCP shutdown")
        self._thread.join(timeout=CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning("Superset MCP client thread did not stop promptly")


@dataclass
class SupersetToolSet:
    """The supported tools and exact schemas reported by a Superset MCP server."""

    client: SupersetMCPClient
    tools: tuple[dict[str, Any], ...]
    missing_tools: tuple[str, ...]
    policy_description: str = ""
    max_mutating_calls: int | None = None
    max_result_chars: int | None = None
    _mutating_call_count: int = field(default=0, init=False, repr=False)
    _mutation_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(tool["name"] for tool in self.tools)

    def catalog_for_model(self) -> str:
        return json.dumps(self.tools, ensure_ascii=False, separators=(",", ":"))

    def restricted(
        self,
        allowed_tool_names: Collection[str],
        *,
        policy_description: str,
        max_mutating_calls: int | None = None,
        max_result_chars: int | None = MAX_SUPERSET_TOOL_RESULT_CHARS,
    ) -> "SupersetToolSet":
        allowed = set(allowed_tool_names)
        return SupersetToolSet(
            client=self.client,
            tools=tuple(tool for tool in self.tools if tool["name"] in allowed),
            missing_tools=self.missing_tools,
            policy_description=policy_description,
            max_mutating_calls=max_mutating_calls,
            max_result_chars=max_result_chars,
        )

    def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self.tool_names:
            return (
                f"Superset MCP tool '{name}' is not available from this server. "
                "Do not claim that the operation was completed."
            )

        if name in MUTATING_SUPERSET_TOOLS and self.max_mutating_calls is not None:
            with self._mutation_lock:
                if self._mutating_call_count >= self.max_mutating_calls:
                    return (
                        "Superset MCP call rejected: this workflow's limit for "
                        "persistent changes has been reached. Do not retry."
                    )
                self._mutating_call_count += 1

        try:
            result = self.client.call_tool(name, arguments)
        except SupersetMCPError as exc:
            return (
                f"Superset MCP request failed: {exc} "
                "The operation was not confirmed as successful."
            )

        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            output = json.dumps(
                structured_content, ensure_ascii=False, default=str
            )
        else:
            pieces: list[str] = []
            for item in getattr(result, "content", ()):
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    pieces.append(text)
                elif getattr(item, "type", None) == "image":
                    pieces.append(
                        "[Superset returned an image preview not rendered in this chat.]"
                    )
            output = "\n".join(pieces) or "Superset MCP returned no textual content."

        if getattr(result, "isError", False):
            output = (
                f"Superset MCP tool error: {output} "
                "Do not report the operation as successful."
            )
        if self.max_result_chars is not None and len(output) > self.max_result_chars:
            output = (
                output[: self.max_result_chars]
                + "\n[Остаток результата Superset скрыт для ограничения объёма контекста.]"
            )
        return output


class SupersetMCPManager:
    """Reuse one MCP client for a given endpoint and credential set."""

    def __init__(self):
        self._lock = threading.Lock()
        self._client: SupersetMCPClient | None = None
        self._client_key: tuple[str, str] | None = None
        self._last_error: str | None = None
        self._last_tool_names: tuple[str, ...] = ()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_tool_names(self) -> tuple[str, ...]:
        return self._last_tool_names

    def _get_client(self, config: SupersetMCPConfig) -> SupersetMCPClient:
        validate_superset_config(config)
        client_key = (config.endpoint_url, config.bearer_token)
        with self._lock:
            if self._client is not None and self._client_key == client_key:
                return self._client
            if self._client is not None:
                self._client.close()
            client = SupersetMCPClient(config)
            self._client = client
            self._client_key = client_key
            return client

    def get_toolset(self, config: SupersetMCPConfig) -> SupersetToolSet:
        """Connect if needed and build a catalog from tools/list metadata."""
        try:
            client = self._get_client(config)
            server_tools = client.list_tools()
            supported = tuple(
                {
                    key: tool[key]
                    for key in ("name", "description", "inputSchema", "annotations")
                    if key in tool
                }
                for tool in server_tools
                if tool.get("name") in SUPPORTED_SUPERSET_TOOLS
            )
            if not supported:
                raise SupersetMCPError(
                    "Соединение установлено, но сервер не предоставил поддерживаемых "
                    "инструментов Superset. Проверьте версию сервера и права пользователя."
                )

            available_names = {tool["name"] for tool in supported}
            missing = tuple(
                name for name in SUPPORTED_SUPERSET_TOOLS if name not in available_names
            )
            self._last_tool_names = tuple(sorted(available_names))
            self._last_error = None
            return SupersetToolSet(client, supported, missing)
        except SupersetMCPError as exc:
            self._last_error = str(exc)
            self._last_tool_names = ()
            raise

    def test_connection(self, config: SupersetMCPConfig) -> dict[str, Any]:
        """Check MCP initialization and report supported tools without exposing secrets."""
        try:
            toolset = self.get_toolset(config)
            health_check_passed = None
            health_tool = next(
                (
                    tool
                    for tool in toolset.tools
                    if tool.get("name") == "health_check"
                ),
                None,
            )
            health_schema = health_tool.get("inputSchema") if health_tool else None
            if (
                isinstance(health_schema, dict)
                and not health_schema.get("required")
            ):
                result = toolset.client.call_tool("health_check", {})
                if getattr(result, "isError", False):
                    raise SupersetMCPError(
                        "MCP-соединение установлено, но health_check Superset вернул ошибку."
                    )
                health_check_passed = True
            return {
                "success": True,
                "tool_count": len(toolset.tools),
                "available_tools": self.last_tool_names,
                "missing_tools": toolset.missing_tools,
                "health_check_passed": health_check_passed,
            }
        except SupersetMCPError as exc:
            self._last_error = str(exc)
            raise

    def close(self) -> None:
        """Close the reusable session, for reconfiguration or app shutdown."""
        with self._lock:
            client = self._client
            self._client = None
            self._client_key = None
            self._last_error = None
            self._last_tool_names = ()
        if client is not None:
            client.close()


superset_mcp_manager = SupersetMCPManager()
