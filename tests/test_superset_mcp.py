import asyncio
import json
import os
import tempfile
import threading
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# AppTest may run the app in an isolated script context.
os.environ["JARVIS_DISABLE_SUPERSET_MCP"] = "true"

import httpx
from streamlit.testing.v1 import AppTest

import config_manager as config_manager_module
from config_manager import ConfigManager, ModelConfig, SupersetMCPConfig
from jarvis_backend import (
    GeminiChat,
    OmnirouteChat,
    _make_superset_model_tool,
    config_manager as app_config_manager,
    create_chat,
)
from superset_mcp import (
    SUPPORTED_SUPERSET_TOOLS,
    SupersetMCPClient,
    SupersetMCPError,
    SupersetMCPManager,
    SupersetToolSet,
    superset_mcp_manager,
    validate_superset_config,
)


class SupersetConfigTests(unittest.TestCase):
    def test_environment_configuration_requires_explicit_enable_and_url(self):
        environment = {
            "SUPERSET_MCP_ENABLED": "true",
            "SUPERSET_MCP_URL": "https://superset.example/mcp-endpoint",
            "SUPERSET_MCP_BEARER_TOKEN": "test-token",
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch("config_manager.CONFIG_FILE", Path(directory) / "missing.json"):
                with patch.dict(os.environ, environment, clear=True):
                    config = ConfigManager().get_superset_mcp_config()

        self.assertTrue(config.enabled)
        self.assertEqual(config.endpoint_url, "https://superset.example/mcp-endpoint")
        self.assertEqual(config.bearer_token, "test-token")

    def test_missing_environment_configuration_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("config_manager.CONFIG_FILE", Path(directory) / "missing.json"):
                with patch.dict(os.environ, {}, clear=True):
                    config = ConfigManager().get_superset_mcp_config()

        self.assertFalse(config.enabled)
        self.assertEqual(config.endpoint_url, "")
        self.assertEqual(config.bearer_token, "")

    def test_environment_kill_switch_disables_saved_configuration(self):
        manager = ConfigManager()
        manager._superset_mcp_config = SupersetMCPConfig(
            enabled=True,
            endpoint_url="https://superset.example/mcp-endpoint",
            bearer_token="test-token",
        )
        with patch.dict(
            os.environ, {"JARVIS_DISABLE_SUPERSET_MCP": "true"}, clear=True
        ):
            config = manager.get_superset_mcp_config()

        self.assertFalse(config.enabled)
        self.assertEqual(config.endpoint_url, "")
        self.assertEqual(config.bearer_token, "")

    def test_saved_configuration_has_restricted_permissions_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / ".jarvis_config.json"
            with patch("config_manager.CONFIG_FILE", config_file), patch.dict(
                os.environ, {}, clear=True
            ):
                manager = ConfigManager()
                manager.set_superset_mcp_config(
                    SupersetMCPConfig(
                        enabled=True,
                        endpoint_url="https://superset.example/mcp-endpoint",
                        bearer_token="test-token",
                    )
                )
                loaded = ConfigManager().get_superset_mcp_config()

            self.assertEqual(config_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(loaded.bearer_token, "test-token")
            self.assertTrue(loaded.enabled)

    def test_endpoint_rejects_remote_plain_http_and_embedded_secrets(self):
        invalid_configs = (
            SupersetMCPConfig(True, "http://superset.example/mcp", ""),
            SupersetMCPConfig(True, "https://user:pass@superset.example/mcp", ""),
            SupersetMCPConfig(True, "https://superset.example/mcp?token=secret", ""),
        )
        for config in invalid_configs:
            with self.subTest(endpoint=config.endpoint_url):
                with self.assertRaises(SupersetMCPError):
                    validate_superset_config(config)

    def test_local_http_endpoint_is_allowed_for_development(self):
        validate_superset_config(
            SupersetMCPConfig(True, "http://localhost:5008/mcp", "")
        )


class SupersetMCPClientTests(unittest.TestCase):
    def test_streamable_http_session_lists_calls_and_closes(self):
        requests = []
        token = "private-test-token"

        def respond(request):
            if request.method == "DELETE":
                requests.append((request, {}))
                return httpx.Response(204)
            if request.method == "GET":
                requests.append((request, {}))
                return httpx.Response(405)
            body = json.loads(request.content)
            requests.append((request, body))
            method = body.get("method")
            if method == "initialize":
                return httpx.Response(
                    200,
                    headers={
                        "content-type": "application/json",
                        "mcp-session-id": "test-session",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "Superset test", "version": "1"},
                        },
                    },
                )
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "health_check",
                                    "description": "Check the MCP server.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {},
                                    },
                                }
                            ]
                        },
                    },
                )
            if method == "tools/call":
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "content": [{"type": "text", "text": "healthy"}]
                        },
                    },
                )
            self.fail(f"Unexpected MCP request: {method}")

        http_clients = []

        def client_factory(*, headers, timeout):
            http_client = httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                transport=httpx.MockTransport(respond),
            )
            http_clients.append(http_client)
            return http_client

        client = SupersetMCPClient(
            SupersetMCPConfig(
                enabled=True,
                endpoint_url="https://superset.example/mcp-endpoint",
                bearer_token=token,
            ),
            httpx_client_factory=client_factory,
        )
        try:
            tools = client.list_tools()
            result = client.call_tool("health_check", {})
        finally:
            client.close()

        self.assertEqual(tools[0]["name"], "health_check")
        self.assertEqual(tools[0]["inputSchema"]["type"], "object")
        self.assertEqual(result.content[0].text, "healthy")
        self.assertEqual(len(http_clients), 1)
        self.assertTrue(http_clients[0].is_closed)
        self.assertTrue(
            any(request.method == "DELETE" for request, _ in requests)
        )
        self.assertTrue(
            all(
                request.headers.get("authorization") == f"Bearer {token}"
                for request, _ in requests
            )
        )

    def test_connection_error_does_not_expose_token(self):
        token = "must-not-be-shown"

        def respond(request):
            return httpx.Response(401, request=request)

        def client_factory(*, headers, timeout):
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                transport=httpx.MockTransport(respond),
            )

        with self.assertRaises(SupersetMCPError) as raised:
            SupersetMCPClient(
                SupersetMCPConfig(
                    enabled=True,
                    endpoint_url="https://superset.example/mcp-endpoint",
                    bearer_token=token,
                ),
                httpx_client_factory=client_factory,
            )

        self.assertIn("401", str(raised.exception))
        self.assertNotIn(token, str(raised.exception))

    def test_tool_timeout_cancels_the_in_flight_mcp_request(self):
        cancelled = threading.Event()

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_value, traceback):
                self.exit_args = (exc_type, exc_value, traceback)
                return None

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_value, traceback):
                self.exit_args = (exc_type, exc_value, traceback)
                return None

            async def initialize(self):
                return None

            async def list_tools(self):
                return SimpleNamespace(tools=[])

            async def call_tool(self, name, arguments):
                self.last_call = (name, arguments)
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        @asynccontextmanager
        async def fake_transport(url, *, http_client, terminate_on_close):
            self.assertEqual(url, "https://superset.example/mcp")
            self.assertIsNotNone(http_client)
            self.assertTrue(terminate_on_close)
            yield object(), object(), lambda: None

        def fake_http_client_factory(*, headers, timeout):
            self.assertIsNone(headers)
            self.assertIsInstance(timeout, httpx.Timeout)
            return FakeHttpClient()

        with patch("superset_mcp.TOOL_TIMEOUT_SECONDS", 0.02), patch(
            "superset_mcp.streamable_http_client", fake_transport
        ), patch("superset_mcp.ClientSession", return_value=FakeSession()):
            client = SupersetMCPClient(
                SupersetMCPConfig(True, "https://superset.example/mcp", ""),
                httpx_client_factory=fake_http_client_factory,
            )
            try:
                with self.assertRaisesRegex(SupersetMCPError, "время ожидания"):
                    client.call_tool("health_check", {})
                self.assertTrue(cancelled.wait(timeout=1))
            finally:
                client.close()


class SupersetToolDiscoveryTests(unittest.TestCase):
    def test_catalog_uses_server_tools_and_reports_missing_capabilities(self):
        server_tool = {
            "name": "list_databases",
            "description": "List accessible databases.",
            "inputSchema": {
                "type": "object",
                "properties": {"search": {"type": "string"}},
            },
        }
        fake_client = MagicMock()
        fake_client.list_tools.return_value = [server_tool, {"name": "unrelated_admin_tool"}]
        manager = SupersetMCPManager()

        with patch("superset_mcp.SupersetMCPClient", return_value=fake_client) as factory:
            first = manager.get_toolset(
                SupersetMCPConfig(True, "https://superset.example/mcp", "")
            )
            second = manager.get_toolset(
                SupersetMCPConfig(True, "https://superset.example/mcp", "")
            )

        self.assertEqual(first.tools[0]["inputSchema"], server_tool["inputSchema"])
        self.assertIn("list_databases", first.tool_names)
        self.assertIn("get_instance_info", first.missing_tools)
        self.assertEqual(second.client, first.client)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(SUPPORTED_SUPERSET_TOOLS), len(set(SUPPORTED_SUPERSET_TOOLS)))
        manager.close()
        fake_client.close.assert_called_once_with()

    def test_connection_check_calls_server_health_tool_when_available(self):
        fake_client = MagicMock()
        fake_client.list_tools.return_value = [
            {
                "name": "health_check",
                "description": "Health check.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        fake_client.call_tool.return_value = SimpleNamespace(isError=False)
        manager = SupersetMCPManager()

        with patch("superset_mcp.SupersetMCPClient", return_value=fake_client):
            result = manager.test_connection(
                SupersetMCPConfig(True, "https://superset.example/mcp", "")
            )

        self.assertTrue(result["health_check_passed"])
        fake_client.call_tool.assert_called_once_with("health_check", {})
        manager.close()

    def test_model_tool_rejects_mutating_sql_before_mcp_call(self):
        fake_toolset = MagicMock(spec=SupersetToolSet)
        model_tool = _make_superset_model_tool(fake_toolset)

        result = model_tool(
            "execute_sql",
            json.dumps({"request": {"database_id": 1, "sql": "DELETE FROM users"}}),
        )

        self.assertIn("read-only SELECT", result)
        fake_toolset.invoke.assert_not_called()

    def test_model_tool_forwards_read_only_sql_and_tool_arguments(self):
        fake_toolset = MagicMock(spec=SupersetToolSet)
        fake_toolset.invoke.return_value = '{"rows": []}'
        model_tool = _make_superset_model_tool(fake_toolset)
        arguments = {"request": {"database_id": 12, "sql": "SELECT 1"}}

        result = model_tool("execute_sql", json.dumps(arguments))

        self.assertEqual(result, '{"rows": []}')
        fake_toolset.invoke.assert_called_once_with("execute_sql", arguments)

    def test_mcp_tool_error_is_returned_without_a_success_claim(self):
        fake_client = MagicMock()
        fake_client.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="Permission denied.")],
            structuredContent=None,
            isError=True,
        )
        toolset = SupersetToolSet(
            client=fake_client,
            tools=(
                {
                    "name": "list_databases",
                    "description": "List databases.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ),
            missing_tools=(),
        )

        result = toolset.invoke("list_databases", {})

        self.assertIn("Permission denied.", result)
        self.assertIn("Do not report the operation as successful.", result)

    def test_restricted_toolset_blocks_other_tools_and_limits_persistent_actions(self):
        client = MagicMock()
        client.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text='{"ok":true}')],
            structuredContent=None,
            isError=False,
        )
        toolset = SupersetToolSet(
            client=client,
            tools=(
                {"name": "add_chart_to_existing_dashboard"},
                {"name": "generate_dashboard"},
            ),
            missing_tools=(),
        ).restricted(
            {"add_chart_to_existing_dashboard"},
            policy_description="Add the selected chart once.",
            max_mutating_calls=1,
        )

        rejected_tool = toolset.invoke("generate_dashboard", {})
        first_action = toolset.invoke("add_chart_to_existing_dashboard", {})
        repeated_action = toolset.invoke("add_chart_to_existing_dashboard", {})

        self.assertIn("not available", rejected_tool)
        self.assertEqual(first_action, '{"ok":true}')
        self.assertIn("limit for persistent changes", repeated_action)
        client.call_tool.assert_called_once_with(
            "add_chart_to_existing_dashboard", {}
        )


class SupersetModelToolTests(unittest.TestCase):
    def setUp(self):
        self.mcp_client = MagicMock()
        self.mcp_client.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text='{"databases":[]}')],
            structuredContent=None,
            isError=False,
        )
        self.toolset = SupersetToolSet(
            client=self.mcp_client,
            tools=(
                {
                    "name": "list_databases",
                    "description": "List accessible databases.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "request": {
                                "type": "object",
                                "properties": {"search": {"type": "string"}},
                            }
                        },
                    },
                },
            ),
            missing_tools=(),
        )

    def test_gemini_receives_superset_tool_and_server_schema(self):
        with patch("jarvis_backend.genai.Client", return_value=MagicMock()):
            chat = GeminiChat(
                "gemini-test",
                "test-key",
                superset_toolset=self.toolset,
            )

        tool_names = [
            getattr(tool, "__name__", None)
            for tool in chat.config.tools
        ]
        self.assertIn("call_superset_mcp_tool", tool_names)
        self.assertIn('"name":"list_databases"', chat.system_instruction)
        self.assertIn('"inputSchema"', chat.system_instruction)
        chat.close()

    @patch("jarvis_backend.requests.post")
    def test_omniroute_receives_catalog_and_forwards_superset_calls(self, post):
        tool_response = MagicMock()
        tool_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "superset-call",
                                "type": "function",
                                "function": {
                                    "name": "call_superset_mcp_tool",
                                    "arguments": json.dumps(
                                        {
                                            "tool_name": "list_databases",
                                            "arguments_json": json.dumps(
                                                {"request": {"search": "sales"}}
                                            ),
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        final_response = MagicMock()
        final_response.json.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "Найдена база данных."}}
            ]
        }
        post.side_effect = [tool_response, final_response]
        chat = OmnirouteChat(
            "test-model",
            "test-key",
            "https://omniroute.example/v1",
            superset_toolset=self.toolset,
        )

        response = chat.send_message("Найди базу sales в Superset")

        self.assertEqual(response.text, "Найдена база данных.")
        request = post.call_args_list[0].kwargs["json"]
        self.assertIn(
            "call_superset_mcp_tool",
            [tool["function"]["name"] for tool in request["tools"]],
        )
        self.assertIn('"inputSchema"', request["messages"][0]["content"])
        self.mcp_client.call_tool.assert_called_once_with(
            "list_databases", {"request": {"search": "sales"}}
        )
        tool_result = post.call_args_list[1].kwargs["json"]["messages"][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertIn('{"databases":[]}', tool_result["content"])

    def test_chat_starts_without_superset_when_mcp_is_unavailable(self):
        model_config = ModelConfig(
            provider="gemini",
            model_name="gemini-test",
            api_key="test-key",
        )
        with patch(
            "jarvis_backend.config_manager.get_model_config",
            return_value=model_config,
        ), patch(
            "jarvis_backend.config_manager.get_superset_mcp_config",
            return_value=SupersetMCPConfig(
                enabled=True,
                endpoint_url="https://superset.example/mcp",
            ),
        ), patch(
            "jarvis_backend.superset_mcp_manager.get_toolset",
            side_effect=SupersetMCPError("Superset offline."),
        ), patch("jarvis_backend.GeminiChat") as gemini_chat:
            chat = create_chat()

        self.assertIs(chat, gemini_chat.return_value)
        self.assertIsNone(gemini_chat.call_args.kwargs["superset_toolset"])
        self.assertIn(
            "Superset offline.",
            gemini_chat.call_args.kwargs["superset_unavailable_reason"],
        )

    def test_create_chat_exposes_only_tools_allowed_by_workflow(self):
        model_config = ModelConfig(
            provider="gemini",
            model_name="gemini-test",
            api_key="test-key",
        )
        server_toolset = SupersetToolSet(
            client=MagicMock(),
            tools=(
                {"name": "list_dashboards"},
                {"name": "generate_dashboard"},
            ),
            missing_tools=(),
        )
        with patch(
            "jarvis_backend.config_manager.get_model_config",
            return_value=model_config,
        ), patch(
            "jarvis_backend.config_manager.get_superset_mcp_config",
            return_value=SupersetMCPConfig(
                enabled=True,
                endpoint_url="https://superset.example/mcp",
            ),
        ), patch(
            "jarvis_backend.superset_mcp_manager.get_toolset",
            return_value=server_toolset,
        ), patch("jarvis_backend.GeminiChat") as gemini_chat:
            create_chat(
                allowed_superset_tools=frozenset({"list_dashboards"}),
                include_database_tool=False,
                max_tool_calls=1,
                max_mutating_calls=0,
                policy_description="Read only.",
                require_superset_tools=True,
            )

        restricted = gemini_chat.call_args.kwargs["superset_toolset"]
        self.assertEqual(restricted.tool_names, frozenset({"list_dashboards"}))
        self.assertEqual(restricted.max_mutating_calls, 0)
        self.assertFalse(gemini_chat.call_args.kwargs["include_database_tool"])


class SupersetStreamlitTests(unittest.TestCase):
    def test_prebuilt_workflow_picker_contains_seven_scenarios(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        with patch.dict(os.environ, {"ALLOW_DB_CONFIG_UI": "false"}, clear=True):
            app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox(key="prebuilt_scenario").options), 7)
        app.selectbox(key="prebuilt_scenario").set_value("superset_add_chart").run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(button.label == "Загрузить дашборды" for button in app.button)
        )

    def test_superset_settings_form_saves_token_without_echoing_it(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                config_manager_module, "CONFIG_FILE", Path(directory) / "config.json"
            ):
                with patch.dict(os.environ, {"ALLOW_DB_CONFIG_UI": "true"}, clear=True):
                    with patch.object(
                        config_manager_module.ConfigManager,
                        "get_superset_mcp_config",
                        return_value=SupersetMCPConfig(),
                    ), patch.object(
                        app_config_manager, "set_superset_mcp_config"
                    ) as save_config, patch.object(
                        superset_mcp_manager, "close"
                    ), patch.object(
                        app_config_manager, "get_database_config", return_value=None
                    ), patch.object(
                        app_config_manager, "get_model_config", return_value=None
                    ), patch.object(
                        app_config_manager, "get_omniroute_client", return_value=None
                    ):
                        app = AppTest.from_file(str(app_path)).run()
                        self.assertFalse(app.exception)

                        app.checkbox(key="superset_mcp_enabled_input").set_value(True)
                        app.text_input(key="superset_mcp_endpoint_input").set_value(
                            "https://superset.example/mcp-endpoint"
                        )
                        app.text_input(key="superset_mcp_token_input").set_value(
                            "private-test-token"
                        )
                        save_button = next(
                            button
                            for button in app.button
                            if button.label == "Сохранить настройки Superset MCP"
                        )
                        save_button.click().run()

                        save_config.assert_called_once()
                        saved = save_config.call_args.args[0]
                        self.assertTrue(saved.enabled)
                        self.assertEqual(saved.bearer_token, "private-test-token")
                        self.assertFalse(app.exception)
                        self.assertEqual(
                            app.text_input(key="superset_mcp_token_input").value, ""
                        )


if __name__ == "__main__":
    unittest.main()
