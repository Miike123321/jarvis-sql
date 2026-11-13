import json
import logging
import os
import re

import pymssql
import streamlit as st

logger = logging.getLogger("jarvis")

from config_manager import (
    DatabaseConfig,
    ModelConfig,
    PLACEHOLDER_API_KEY,
    SupersetMCPConfig,
)
from jarvis_backend import (
    GEMINI_MODEL,
    GeminiChat,
    GeminiResponseError,
    MAX_MESSAGE_LENGTH,
    MAX_OMNIROUTE_TOOL_CALLS,
    MAX_OMNIROUTE_TOOL_ROUNDS,
    create_chat,
    database_config,
    execute_ddl_statement,
    is_approved_ddl,
    config_manager,
    test_model_connection,
)
from superset_mcp import (
    SUPERSET_REQUESTED_PERMISSIONS,
    SupersetMCPError,
    superset_mcp_manager,
)


SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", re.IGNORECASE)
ASSET_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
REQUIRED_SETTINGS = (
    "GEMINI_API_KEY",
    "DB_HOST",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
)
READ_ONLY_SUPERSET_TOOLS = frozenset(
    {
        "health_check",
        "get_instance_info",
        "get_schema",
        "list_databases",
        "get_database_info",
        "list_datasets",
        "get_dataset_info",
        "query_dataset",
        "list_charts",
        "get_chart_info",
        "get_chart_data",
        "get_chart_preview",
        "get_chart_sql",
        "get_chart_type_schema",
        "generate_explore_link",
        "list_dashboards",
        "get_dashboard_info",
        "get_dashboard_data",
        "execute_sql",
        "open_sql_lab_with_context",
    }
)
PREBUILT_SCENARIOS = {
    "sql_question": "SQL: ответить на вопрос по данным",
    "sql_table": "SQL: проверить структуру таблицы",
    "sql_compare": "SQL: сравнить показатели",
    "superset_find": "Superset: найти доступные данные и объекты",
    "superset_inspect": "Superset: изучить график, дашборд или датасет",
    "superset_add_chart": "Superset: добавить график в выбранный дашборд",
    "superset_create_dashboard": "Superset: создать дашборд по шаблону",
}

st.set_page_config(
    page_title="J.A.R.V.I.S. SQL Assistant",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 50% 0, #12253a 0, transparent 48%),
            #080d16;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #26364b;
        border-radius: 14px;
        background: #111c2a;
    }
    [data-testid="stChatMessage"] pre {
        border: 1px solid #23536a;
        border-left: 3px solid #28b8dd;
        background: #080d16;
        color: #9ce9bd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session() -> None:
    defaults = {
        "chat": None,
        "messages": [],
        "queued_messages": [],
        "pending_ddl": None,
        "chat_error": None,
        "model_tested": False,
        "model_test_passed": False,
        "_last_saved_provider": "",
        "_last_saved_model_id": "",
        "_loaded_models": None,
        "_model_options": None,
        "_model_selector_shown": False,
        "_last_chat_provider": None,
        "preset_dashboard_choices": {},
        "preset_chart_choices": {},
        "preset_load_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_model_session() -> None:
    chat = st.session_state["chat"]
    if isinstance(chat, GeminiChat):
        chat.close()
    st.session_state["chat"] = None
    st.session_state["chat_error"] = None
    st.session_state["messages"] = []
    st.session_state["queued_messages"] = []
    st.session_state["pending_ddl"] = None
    st.session_state["model_test_passed"] = False


def queue_draft() -> None:
    text = st.session_state.get("draft_message", "").strip()
    if not text:
        return
    if len(text) > MAX_MESSAGE_LENGTH:
        st.session_state["draft_error"] = (
            f"Одна часть запроса не должна превышать {MAX_MESSAGE_LENGTH} символов."
        )
        return
    st.session_state["queued_messages"].append(text)
    st.session_state["draft_message"] = ""
    st.session_state["draft_error"] = None


def extract_superset_asset_choices(
    payload: object,
    *,
    id_keys: tuple[str, ...],
    label_keys: tuple[str, ...],
) -> dict[str, str]:
    """Extract selectable asset IDs and labels from a Superset MCP result."""
    choices: dict[str, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            identifier = next(
                (
                    value[key]
                    for key in id_keys
                    if key in value and isinstance(value[key], (str, int))
                ),
                None,
            )
            label = next(
                (
                    value[key]
                    for key in label_keys
                    if key in value and isinstance(value[key], str) and value[key].strip()
                ),
                None,
            )
            if identifier is not None and label is not None:
                choices[str(identifier)] = f"{label.strip()} (ID: {identifier})"
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return choices


def load_superset_asset_choices(tool_name: str, asset_kind: str) -> dict[str, str]:
    """Load a dropdown list only when the MCP tool can be called without guessed arguments."""
    toolset = superset_mcp_manager.get_toolset(
        config_manager.get_superset_mcp_config()
    )
    metadata = next(
        (tool for tool in toolset.tools if tool.get("name") == tool_name), None
    )
    if metadata is None:
        raise SupersetMCPError(
            f"Сервер Superset не предоставляет инструмент {tool_name}."
        )

    schema = metadata.get("inputSchema")
    required = schema.get("required", []) if isinstance(schema, dict) else []
    if required:
        raise SupersetMCPError(
            f"Нельзя безопасно загрузить список: {tool_name} требует параметры "
            "из схемы сервера. Используйте свободный чат для просмотра объектов."
        )

    result_text = toolset.invoke(tool_name, {})
    try:
        payload = json.loads(result_text)
    except json.JSONDecodeError as exc:
        raise SupersetMCPError(
            f"Не удалось разобрать список {asset_kind}, полученный от Superset."
        ) from exc

    if asset_kind == "дашбордов":
        choices = extract_superset_asset_choices(
            payload,
            id_keys=("dashboard_id", "id", "uuid"),
            label_keys=("dashboard_title", "title", "name"),
        )
    else:
        choices = extract_superset_asset_choices(
            payload,
            id_keys=("chart_id", "id", "uuid"),
            label_keys=("chart_title", "slice_name", "title", "name"),
        )
    if not choices:
        raise SupersetMCPError(
            f"Superset не вернул идентификаторы и названия {asset_kind}."
        )
    return choices


def submit_prebuilt_scenario(
    prompt: str,
    *,
    allowed_superset_tools: frozenset[str] = frozenset(),
    include_database_tool: bool = True,
    max_tool_calls: int = 2,
    max_tool_rounds: int = 2,
    max_mutating_calls: int | None = 0,
    policy_description: str = "",
) -> None:
    """Run one isolated, tool-scoped workflow and append its result to the chat."""
    scenario_chat = create_chat(
        allowed_superset_tools=allowed_superset_tools,
        include_database_tool=include_database_tool,
        max_tool_calls=max_tool_calls,
        max_tool_rounds=max_tool_rounds,
        max_mutating_calls=max_mutating_calls,
        policy_description=policy_description,
        require_superset_tools=bool(allowed_superset_tools),
    )
    try:
        response = scenario_chat.send_message(prompt)
        answer = getattr(response, "text", None)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Модель не вернула текстовый ответ.")
    finally:
        close = getattr(scenario_chat, "close", None)
        if callable(close):
            close()

    st.session_state["messages"].extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )


def show_message(message: dict[str, str]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            st.text(message["content"])


def show_config_ui() -> None:
    """Display configuration management UI in sidebar."""
    if not config_manager.is_ui_config_enabled():
        return

    with st.expander("⚙️ Настройки конфигурации", expanded=False):
        st.subheader("База данных")

        # Get current DB config
        current_db = config_manager.get_database_config()

        db_host = st.text_input(
            "Хост БД",
            value=current_db.host if current_db else "",
            key="db_host_input",
        )
        db_port = st.number_input(
            "Порт БД",
            value=current_db.port if current_db else 1433,
            min_value=1,
            max_value=65535,
            key="db_port_input",
        )
        db_name = st.text_input(
            "Имя БД",
            value=current_db.name if current_db else "",
            key="db_name_input",
        )
        db_user = st.text_input(
            "Пользователь БД",
            value=current_db.user if current_db else "",
            key="db_user_input",
        )
        db_password = st.text_input(
            "Пароль БД",
            value=current_db.password if current_db else "",
            type="password",
            key="db_password_input",
        )

        if st.button("💾 Сохранить настройки БД", use_container_width=True):
            if all([db_host, db_name, db_user, db_password]):
                try:
                    new_db_config = DatabaseConfig(
                        host=db_host,
                        port=int(db_port),
                        name=db_name,
                        user=db_user,
                        password=db_password,
                    )
                    config_manager.set_database_config(new_db_config)
                    st.success("✅ Настройки БД сохранены!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
            else:
                st.error("❌ Все поля обязательны")

        st.divider()
        st.subheader("Модель LLM")

        current_model = config_manager.get_model_config()
        default_provider = (
            current_model.provider
            if current_model and current_model.provider in {"omniroute", "gemini"}
            else "omniroute"
        )

        model_provider = st.radio(
            "Выберите модель:",
            options=["omniroute", "gemini"],
            horizontal=True,
            index=0 if default_provider == "omniroute" else 1,
            key="model_provider_select",
            label_visibility="collapsed",
        )

        if model_provider == "omniroute":
            st.caption("🤖 AI через локальный сервер omniroute")
            st.info("Omniroute выполняет запросы к базе данных только через безопасный SELECT-инструмент.")
            omniroute_client = config_manager.get_omniroute_client()

            if not omniroute_client:
                st.warning("⚠️ Настройте OMNIROUTE_API_KEY в файле .env.")
            else:
                if current_model and current_model.provider == "omniroute":
                    st.success(f"✅ Используется: **{current_model.model_name}**")

                if st.button("🔄 Загрузить доступные модели", use_container_width=True):
                    with st.spinner("Загружаю модели..."):
                        models = omniroute_client.get_available_models()
                    st.session_state["_loaded_models"] = [
                        model
                        for model in models
                        if isinstance(model.get("id"), str) and model["id"]
                    ]

                models = st.session_state.get("_loaded_models") or []
                model_options = {
                    f"{model['id']} - {model.get('name', 'Unknown')}": model["id"]
                    for model in models
                }
                if model_options:
                    selected_display = st.selectbox(
                        "Выберите модель",
                        options=list(model_options),
                        key="omniroute_model_select",
                    )
                    selected_model_id = model_options[selected_display]
                    if st.button("Использовать выбранную модель", use_container_width=True):
                        new_model_config = ModelConfig(
                            provider="omniroute",
                            model_name=selected_model_id,
                            api_key=omniroute_client.api_key,
                            base_url=omniroute_client.base_url,
                        )
                        if new_model_config != current_model:
                            try:
                                config_manager.set_model_config(new_model_config)
                                reset_model_session()
                                st.toast(
                                    f"Переключено на {selected_model_id}",
                                    icon="🎉",
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"❌ Не удалось сохранить модель: {exc}")
                elif st.session_state.get("_loaded_models") is not None:
                    st.error("❌ Не удалось загрузить модели. Проверьте адрес и API-ключ.")

        else:  # gemini provider
            st.caption("💎 Google Gemini API")
            if current_model and current_model.provider == "gemini":
                st.success(f"✅ Используется: **{current_model.model_name}**")

            with st.form("gemini_api_key_form", clear_on_submit=True):
                gemini_api_key = st.text_input(
                    "Gemini API-ключ",
                    type="password",
                    placeholder="Вставьте ключ Google AI Studio",
                    key="gemini_api_key_input",
                )
                save_gemini_key = st.form_submit_button(
                    "Сохранить ключ Gemini",
                    use_container_width=True,
                )

            if save_gemini_key:
                gemini_api_key = gemini_api_key.strip()
                if not gemini_api_key or gemini_api_key == PLACEHOLDER_API_KEY:
                    st.error("❌ Введите действительный Gemini API-ключ.")
                else:
                    model_name = (
                        current_model.model_name
                        if current_model and current_model.provider == "gemini"
                        else os.getenv("GEMINI_MODEL", "").strip() or GEMINI_MODEL
                    )
                    new_model_config = ModelConfig(
                        provider="gemini",
                        model_name=model_name,
                        api_key=gemini_api_key,
                    )
                    try:
                        config_manager.set_model_config(new_model_config)
                        reset_model_session()
                    except Exception as exc:
                        st.error(f"❌ Не удалось сохранить модель: {exc}")
                    else:
                        st.success("✅ Gemini API-ключ сохранён.")
                        st.rerun()

        st.divider()
        st.subheader("Apache Superset MCP")
        current_superset = config_manager.get_superset_mcp_config()
        with st.expander("Права для пользователя Superset"):
            st.caption(
                "Это RBAC-права Superset, а не MCP-инструменты. Попросите "
                "администратора назначить их роли/пользователю, под которым "
                "работает MCP-сервер. Jarvis не меняет права Superset. "
                "Выдавайте write- и data-access права только в нужном объёме."
            )
            st.code("\n".join(SUPERSET_REQUESTED_PERMISSIONS))
            st.caption(
                "`can_read on Chart` указано вместо опечатки `an_read on Chart`. "
                "Наличие `can_format_sql` и `can_estimate_query_cost` зависит от "
                "версии и конфигурации Superset."
            )
        if current_superset.bearer_token:
            st.caption("Bearer-токен сохранён; оставьте поле пустым, чтобы не менять его.")

        def save_superset_configuration() -> None:
            current = config_manager.get_superset_mcp_config()
            enabled = st.session_state.get("superset_mcp_enabled_input", False)
            endpoint = st.session_state.get("superset_mcp_endpoint_input", "").strip()
            if enabled and not endpoint:
                st.session_state["superset_mcp_save_error"] = (
                    "Укажите endpoint MCP-сервера Superset."
                )
                return

            entered_token = st.session_state.get("superset_mcp_token_input", "").strip()
            bearer_token = (
                ""
                if st.session_state.get("superset_mcp_remove_token_input", False)
                else entered_token or current.bearer_token
            )
            try:
                config_manager.set_superset_mcp_config(
                    SupersetMCPConfig(
                        enabled=enabled,
                        endpoint_url=endpoint,
                        bearer_token=bearer_token,
                    )
                )
                superset_mcp_manager.close()
                st.session_state["preset_dashboard_choices"] = {}
                st.session_state["preset_chart_choices"] = {}
                reset_model_session()
            except Exception as exc:
                st.session_state["superset_mcp_save_error"] = (
                    f"Не удалось сохранить настройки Superset MCP: {exc}"
                )
            else:
                st.session_state["superset_mcp_save_success"] = True
                st.session_state["superset_mcp_token_input"] = ""

        with st.form("superset_mcp_config_form", clear_on_submit=True):
            st.checkbox(
                "Включить интеграцию Superset MCP",
                value=current_superset.enabled,
                key="superset_mcp_enabled_input",
            )
            st.text_input(
                "Streamable HTTP MCP endpoint",
                value=current_superset.endpoint_url,
                placeholder="Введите точный URL, предоставленный администратором Superset",
                key="superset_mcp_endpoint_input",
            )
            st.text_input(
                "Bearer-токен (если требуется сервером)",
                type="password",
                placeholder="Оставьте пустым, чтобы сохранить текущий",
                key="superset_mcp_token_input",
            )
            st.checkbox(
                "Удалить сохранённый Bearer-токен",
                value=False,
                key="superset_mcp_remove_token_input",
            )
            st.form_submit_button(
                "Сохранить настройки Superset MCP",
                use_container_width=True,
                on_click=save_superset_configuration,
            )

        save_error = st.session_state.pop("superset_mcp_save_error", None)
        if save_error:
            st.error(save_error)
        if st.session_state.pop("superset_mcp_save_success", False):
            st.success("Настройки Superset MCP сохранены.")

        if st.button(
            "Проверить подключение к Superset MCP",
            use_container_width=True,
            key="test_superset_mcp_connection",
        ):
            try:
                with st.spinner("Проверяю MCP endpoint и доступные инструменты..."):
                    result = superset_mcp_manager.test_connection(
                        config_manager.get_superset_mcp_config()
                    )
                st.success(
                    f"Соединение установлено; доступно инструментов Superset: "
                    f"{result['tool_count']}."
                )
                if result["health_check_passed"]:
                    st.caption("Проверка health_check Superset прошла успешно.")
                if result["missing_tools"]:
                    st.info(
                        "Часть возможностей недоступна на этом сервере или для "
                        "этого пользователя: "
                        + ", ".join(result["missing_tools"])
                    )
            except SupersetMCPError as exc:
                st.error(str(exc))

        if not current_superset.enabled:
            st.caption("Интеграция отключена.")
        elif superset_mcp_manager.last_error:
            st.warning(superset_mcp_manager.last_error)
        elif superset_mcp_manager.last_tool_names:
            st.success(
                f"Подключено; доступно инструментов: "
                f"{len(superset_mcp_manager.last_tool_names)}."
            )
        else:
            st.caption("Подключение будет проверено при вызове инструментов.")


def show_prebuilt_workflows() -> None:
    """Render bounded SQL/Superset workflows alongside the unrestricted text box."""
    st.divider()
    st.subheader("Готовые сценарии")
    st.caption(
        "Сценарии отправляют короткий запрос в отдельном контексте и открывают "
        "только нужные инструменты. В свободном чате Superset доступен только "
        "для чтения; постоянные изменения запускаются через подтверждённые формы."
    )
    scenario = st.selectbox(
        "Выберите задачу",
        options=list(PREBUILT_SCENARIOS),
        format_func=PREBUILT_SCENARIOS.get,
        key="prebuilt_scenario",
    )

    if scenario == "superset_add_chart":
        refresh_columns = st.columns(2)
        if refresh_columns[0].button(
            "Загрузить дашборды",
            key="refresh_preset_dashboards",
            use_container_width=True,
        ):
            try:
                st.session_state["preset_dashboard_choices"] = (
                    load_superset_asset_choices("list_dashboards", "дашбордов")
                )
                st.session_state["preset_load_error"] = None
            except SupersetMCPError as exc:
                st.session_state["preset_load_error"] = str(exc)
        if refresh_columns[1].button(
            "Загрузить графики",
            key="refresh_preset_charts",
            use_container_width=True,
        ):
            try:
                st.session_state["preset_chart_choices"] = (
                    load_superset_asset_choices("list_charts", "графиков")
                )
                st.session_state["preset_load_error"] = None
            except SupersetMCPError as exc:
                st.session_state["preset_load_error"] = str(exc)
        if st.session_state.get("preset_load_error"):
            st.error(st.session_state["preset_load_error"])

    prompt = ""
    allowed_tools = frozenset()
    include_database_tool = True
    max_tool_calls = 2
    max_tool_rounds = 2
    max_mutating_calls = 0
    policy = "Выполни только выбранную задачу; не создавай и не изменяй объекты."
    confirmed = True
    valid = True

    with st.form(f"prebuilt_form_{scenario}"):
        if scenario == "sql_question":
            question = st.text_area(
                "Что нужно узнать из базы?",
                max_chars=900,
                key="preset_sql_question",
                placeholder="Например: сколько заказов было по месяцам за последний год?",
            ).strip()
            valid = bool(question)
            prompt = (
                "Ответь на вопрос по настроенной Microsoft SQL Server. "
                "Используй только read-only SELECT и выполни максимум один запрос. "
                "Не предлагай DDL или другие изменения. Вопрос пользователя:\n"
                + question
            )
            max_tool_calls = 1
            max_tool_rounds = 1
        elif scenario == "sql_table":
            schema_name = st.text_input(
                "Схема таблицы",
                value="dbo",
                max_chars=128,
                key="preset_sql_schema",
            ).strip()
            table_name = st.text_input(
                "Таблица",
                max_chars=128,
                key="preset_sql_table",
                placeholder="Например: Orders",
            ).strip()
            valid = bool(
                ASSET_IDENTIFIER.fullmatch(schema_name)
                and ASSET_IDENTIFIER.fullmatch(table_name)
            )
            prompt = (
                f"Проверь структуру таблицы {schema_name}.{table_name} в настроенной "
                "Microsoft SQL Server. Используй максимум один безопасный SELECT "
                "к INFORMATION_SCHEMA или sys-метаданным. Ничего не изменяй."
            )
            max_tool_calls = 1
            max_tool_rounds = 1
        elif scenario == "sql_compare":
            comparison = st.text_area(
                "Какие показатели сравнить?",
                max_chars=900,
                key="preset_sql_compare",
                placeholder="Например: сравни выручку и количество заказов по кварталам.",
            ).strip()
            valid = bool(comparison)
            prompt = (
                "Сравни показатели в настроенной Microsoft SQL Server. Сначала "
                "найди подходящие таблицы только read-only SELECT-запросами, затем "
                "выполни сравнение; максимум два вызова SQL-инструмента. "
                "Не предлагай и не выполняй изменения.\nЗадача:\n"
                + comparison
            )
            max_tool_calls = 2
            max_tool_rounds = 2
        elif scenario == "superset_find":
            search = st.text_input(
                "Что найти в Superset?",
                max_chars=500,
                key="preset_superset_search",
                placeholder="Например: датасеты и дашборды по продажам",
            ).strip()
            valid = bool(search)
            allowed_tools = frozenset(
                {"list_databases", "list_datasets", "list_charts", "list_dashboards"}
            )
            include_database_tool = False
            max_tool_calls = 3
            max_tool_rounds = 3
            policy = "Только поиск/просмотр. Не выполняй SQL и не изменяй объекты."
            prompt = (
                "Найди в Superset доступные базы, датасеты, графики или дашборды "
                "по теме ниже. Используй только результаты инструментов, не "
                "запускай SQL и не создавай объекты. Запрос:\n" + search
            )
        elif scenario == "superset_inspect":
            object_kind = st.selectbox(
                "Тип объекта",
                options=["график", "дашборд", "датасет"],
                key="preset_superset_object_kind",
            )
            object_reference = st.text_input(
                "Название или ID объекта",
                max_chars=300,
                key="preset_superset_object_reference",
            ).strip()
            valid = bool(object_reference)
            allowed_tools = frozenset(
                {
                    "list_charts",
                    "get_chart_info",
                    "get_chart_data",
                    "get_chart_preview",
                    "get_chart_sql",
                    "list_dashboards",
                    "get_dashboard_info",
                    "get_dashboard_data",
                    "list_datasets",
                    "get_dataset_info",
                    "get_schema",
                }
            )
            include_database_tool = False
            max_tool_calls = 3
            max_tool_rounds = 3
            policy = "Только просмотр выбранного объекта; ничего не сохраняй и не изменяй."
            prompt = (
                f"Изучи {object_kind} Superset с названием или ID "
                f"{object_reference!r}. Сначала проверь доступность и идентификатор, "
                "затем верни краткую сводку из доступных данных. Не запускай SQL "
                "и не изменяй объект."
            )
        elif scenario == "superset_add_chart":
            dashboards = st.session_state["preset_dashboard_choices"]
            charts = st.session_state["preset_chart_choices"]
            dashboard_ids = list(dashboards) or [""]
            chart_ids = list(charts) or [""]
            dashboard_id = st.selectbox(
                "Целевой дашборд",
                options=dashboard_ids,
                format_func=lambda value: dashboards.get(
                    value, "Сначала загрузите список дашбордов"
                ),
                key="preset_dashboard_id",
            )
            chart_id = st.selectbox(
                "График для добавления",
                options=chart_ids,
                format_func=lambda value: charts.get(
                    value, "Сначала загрузите список графиков"
                ),
                key="preset_chart_id",
            )
            confirmed = st.checkbox(
                "Подтверждаю добавление выбранного графика в этот дашборд",
                key="confirm_preset_add_chart",
            )
            valid = bool(dashboards and charts and dashboard_id and chart_id)
            allowed_tools = frozenset({"add_chart_to_existing_dashboard"})
            include_database_tool = False
            max_tool_calls = 1
            max_tool_rounds = 1
            max_mutating_calls = 1
            policy = (
                "Выполни ровно одно подтверждённое действие: добавь выбранный "
                "график в выбранный существующий дашборд. Не создавай дашборд."
            )
            prompt = (
                f"Добавь существующий график с ID {chart_id!r} в существующий "
                f"дашборд с ID {dashboard_id!r}. Используй исключительно "
                "add_chart_to_existing_dashboard. Не создавай новый дашборд и "
                "не меняй другие объекты."
            )
        else:
            dashboard_title = st.text_input(
                "Название нового дашборда",
                max_chars=120,
                key="preset_dashboard_title",
            ).strip()
            dashboard_theme = st.selectbox(
                "Шаблон содержания",
                options=[
                    "Руководительские KPI",
                    "Продажи и воронка",
                    "Финансы и маржинальность",
                    "Операции и SLA",
                    "Клиенты и удержание",
                ],
                key="preset_dashboard_theme",
            )
            dashboard_goal = st.text_area(
                "Какие вопросы должен закрывать дашборд?",
                max_chars=600,
                key="preset_dashboard_goal",
                placeholder="Укажите нужные показатели, период и аудиторию.",
            ).strip()
            confirmed = st.checkbox(
                "Подтверждаю создание одного постоянного дашборда",
                key="confirm_preset_create_dashboard",
            )
            valid = bool(dashboard_title and dashboard_goal)
            allowed_tools = frozenset(
                {
                    "list_datasets",
                    "get_dataset_info",
                    "list_charts",
                    "generate_dashboard",
                }
            )
            include_database_tool = False
            max_tool_calls = 4
            max_tool_rounds = 4
            max_mutating_calls = 1
            policy = (
                "Сначала используй доступные существующие датасеты/графики; затем "
                "создай ровно один дашборд по подтверждённому шаблону. Не создавай "
                "графики, virtual datasets или другие объекты."
            )
            prompt = (
                f"Создай один дашборд с названием {dashboard_title!r}. "
                f"Шаблон содержания: {dashboard_theme}. Цель и показатели: "
                f"{dashboard_goal}. Сначала проверь доступные датасеты и графики; "
                "используй только то, что вернули инструменты. Создание уже "
                "подтверждено пользователем. Не создавай другие объекты."
            )

        submitted = st.form_submit_button(
            "Запустить выбранный сценарий",
            type="primary",
            disabled=not st.session_state.get("model_test_passed", False),
        )

    if submitted:
        if not valid:
            st.error("Заполните обязательные поля корректными значениями.")
        elif not confirmed:
            st.warning("Для постоянного изменения отметьте явное подтверждение.")
        elif len(prompt) > MAX_MESSAGE_LENGTH:
            st.error(f"Сценарий превышает лимит {MAX_MESSAGE_LENGTH} символов.")
        elif scenario.startswith("sql_") and not config_manager.get_database_config():
            st.error("Настройте подключение к Microsoft SQL Server.")
        else:
            try:
                with st.spinner("Выполняю ограниченный сценарий..."):
                    submit_prebuilt_scenario(
                        prompt,
                        allowed_superset_tools=allowed_tools,
                        include_database_tool=include_database_tool,
                        max_tool_calls=max_tool_calls,
                        max_tool_rounds=max_tool_rounds,
                        max_mutating_calls=max_mutating_calls,
                        policy_description=policy,
                    )
                st.rerun()
            except Exception as exc:
                logger.exception("Prebuilt workflow failed")
                st.error(f"Сценарий не выполнен: {exc}")


initialize_session()

st.title("⚡ J.A.R.V.I.S.")
st.caption("SQL-ассистент · AI + MS SQL")

with st.sidebar:
    st.subheader("Состояние")

    # Check configuration status for button disable
    db_configured = config_manager.get_database_config() is not None
    model_configured = config_manager.get_model_config() is not None

    if not (db_configured and model_configured):
        st.warning("⚠️ Заполните конфигурацию выше или установите переменные .env")
    else:
        st.success("✅ Конфигурация готова")

        current_model = config_manager.get_model_config()
        if current_model:
            model_label = f"{current_model.provider} ({current_model.model_name})"
            st.caption(f"Модель: `{model_label}`")
        else:
            st.caption(f"Модель: `{GEMINI_MODEL}`")

    # Show configuration UI
    show_config_ui()

    st.divider()

    # Check if we have config from ConfigManager
    db_configured = config_manager.get_database_config() is not None
    model_configured = config_manager.get_model_config() is not None

    if not (db_configured and model_configured):
        st.warning("⚠️ Заполните конфигурацию выше или установите переменные .env")
    else:
        st.success("✅ Конфигурация готова")

    if st.button("Проверить подключение к MS SQL Server", use_container_width=True):
        connection = None
        try:
            with st.spinner("Проверяю подключение..."):
                connection = pymssql.connect(**database_config())
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            st.success("✅ Подключение к MS SQL Server работает")
        except Exception as e:
            st.error(f"❌ Не удалось подключиться: {e}")
        finally:
            if connection is not None:
                connection.close()

    if st.button("🧪 Проверить модель LLM", use_container_width=True):
        with st.spinner("Тестирую модель..."):
            result = test_model_connection()

        if result["success"]:
            st.success(f"✅ Модель работает! Ответ: {result.get('answer', '')}")
            st.session_state["model_test_passed"] = True
        else:
            st.error(f"❌ Ошибка модели: {result.get('error', 'Неизвестная ошибка')}")
            st.session_state["model_test_passed"] = False

    st.divider()
    st.caption(
        "Сообщения копятся в черновике. Ассистент получит их только после нажатия "
        "«Отправить ассистенту»."
    )

if st.session_state["chat"] is None and (db_configured and model_configured):
    try:
        # Check if provider changed - reset chat if needed
        current_provider = config_manager.get_model_config().provider if config_manager.get_model_config() else None
        saved_provider = st.session_state.get("_last_chat_provider", None)

        if current_provider != saved_provider:
            logger.info(f"Provider changed from {saved_provider} to {current_provider}, resetting chat")
            st.session_state["chat"] = create_chat(
                allowed_superset_tools=READ_ONLY_SUPERSET_TOOLS,
                max_tool_calls=MAX_OMNIROUTE_TOOL_CALLS,
                max_tool_rounds=MAX_OMNIROUTE_TOOL_ROUNDS,
                max_mutating_calls=0,
                policy_description=(
                    "Это свободный чат Streamlit с доступом Superset только для "
                    "чтения. Инструменты постоянных изменений недоступны; "
                    "предложи пользователю соответствующий готовый сценарий."
                ),
            )
            st.session_state["_last_chat_provider"] = current_provider
        else:
            st.session_state["chat"] = create_chat(
                allowed_superset_tools=READ_ONLY_SUPERSET_TOOLS,
                max_tool_calls=MAX_OMNIROUTE_TOOL_CALLS,
                max_tool_rounds=MAX_OMNIROUTE_TOOL_ROUNDS,
                max_mutating_calls=0,
                policy_description=(
                    "Это свободный чат Streamlit с доступом Superset только для "
                    "чтения. Инструменты постоянных изменений недоступны; "
                    "предложи пользователю соответствующий готовый сценарий."
                ),
            )
    except Exception as exc:
        st.session_state["chat_error"] = str(exc)

superset_status = config_manager.get_superset_mcp_config()
if superset_status.enabled and superset_mcp_manager.last_error:
    st.warning(
        "Superset MCP сейчас недоступен; обычный чат продолжает работать без "
        f"этой интеграции. {superset_mcp_manager.last_error}"
    )
elif superset_status.enabled and superset_mcp_manager.last_tool_names:
    st.info(
        f"Superset MCP подключён; доступно инструментов: "
        f"{len(superset_mcp_manager.last_tool_names)}."
    )

if st.session_state["chat_error"]:
    st.error(st.session_state["chat_error"])

for saved_message in st.session_state["messages"]:
    show_message(saved_message)

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Здравствуйте! Добавьте одну или несколько частей задачи, "
            "затем отправьте их одним запросом. Для типовых операций используйте "
            "готовые сценарии ниже."
        )

show_prebuilt_workflows()

st.subheader("Черновик запроса")
with st.form("draft_form", clear_on_submit=False):
    st.text_area(
        "Напишите часть задачи",
        key="draft_message",
        height=120,
        placeholder="Например: сначала опиши нужную таблицу, затем добавь условия и формат результата...",
        label_visibility="collapsed",
    )
    st.form_submit_button(
        "Добавить часть",
        on_click=queue_draft,
        use_container_width=False,
    )

if st.session_state.get("draft_error"):
    st.warning(st.session_state["draft_error"])

queued_messages = st.session_state["queued_messages"]
if queued_messages:
    st.markdown("**Части, которые будут отправлены вместе:**")
    for index, queued_message in enumerate(queued_messages, start=1):
        with st.container(border=True):
            st.markdown(f"**Часть {index}**")
            st.text(queued_message)

    remove_index = st.selectbox(
        "Убрать часть из черновика",
        options=list(range(len(queued_messages))),
        format_func=lambda index: f"Часть {index + 1}",
        key="remove_queue_index",
    )
    remove_col, clear_col, send_col = st.columns([1, 1, 2])
    if remove_col.button("Убрать выбранную", use_container_width=True):
        queued_messages.pop(remove_index)
        st.rerun()
    if clear_col.button("Очистить черновик", use_container_width=True):
        st.session_state["queued_messages"] = []
        st.rerun()

    # Check configuration status for button disable
    db_configured = config_manager.get_database_config() is not None
    model_configured = config_manager.get_model_config() is not None
    config_ready = db_configured and model_configured
    model_tested = st.session_state.get("model_test_passed", False)

    send_disabled = (
        st.session_state["chat"] is None
        or not config_ready
        or not model_tested
        or bool(st.session_state["pending_ddl"])
    )
    if send_col.button(
        "Отправить ассистенту",
        type="primary",
        disabled=send_disabled,
        use_container_width=True,
    ):
        combined_prompt = "\n\n".join(
            f"Дополнение {index}:\n{text}"
            for index, text in enumerate(queued_messages, start=1)
        )
        if len(combined_prompt) > MAX_MESSAGE_LENGTH:
            st.error(
                f"Общий запрос слишком длинный. Лимит — {MAX_MESSAGE_LENGTH} символов."
            )
        else:
            try:
                logger.info(f"Sending message with model config: {config_manager.get_model_config().to_dict() if config_manager.get_model_config() else 'None'}")
                with st.spinner("J.A.R.V.I.S. анализирует задачу..."):
                    response = st.session_state["chat"].send_message(combined_prompt)
                answer = getattr(response, "text", None)
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("Модель не вернула текстовый ответ.")
                logger.info("Response received, text length: %s", len(answer))
                st.session_state["messages"].extend(
                    [
                        {"role": "user", "content": combined_prompt},
                        {"role": "assistant", "content": answer},
                    ]
                )
                sql_match = SQL_FENCE.search(answer)
                if sql_match:
                    proposed_sql = sql_match.group(1).strip()
                    if is_approved_ddl(proposed_sql):
                        st.session_state["pending_ddl"] = proposed_sql
                    else:
                        st.warning(
                            "SQL показан для ознакомления, но его нельзя выполнить: "
                            "разрешены только одиночные CREATE TABLE, VIEW или INDEX."
                        )
                st.session_state["queued_messages"] = []
                st.rerun()
            except GeminiResponseError as e:
                st.error(str(e))
            except Exception as e:
                logger.exception("Model request failed")
                model_config = config_manager.get_model_config()
                if model_config and model_config.provider == "omniroute":
                    provider_name = "omniroute"
                else:
                    provider_name = "Gemini"

                st.error(
                    f"Не удалось получить ответ от {provider_name}.\n\n"
                    f"Ошибка: {str(e)}"
                )

if st.session_state["pending_ddl"]:
    st.divider()
    st.subheader("Подтверждение SQL")
    st.warning("Проверьте запрос. Он изменит базу только после вашего подтверждения.")
    st.code(st.session_state["pending_ddl"], language="sql")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button(
        "Подтвердить и выполнить",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Выполняю подтверждённый SQL..."):
            result = execute_ddl_statement(st.session_state["pending_ddl"])
        st.session_state["messages"].append(
            {"role": "assistant", "content": f"**Результат выполнения SQL:**\n\n{result}"}
        )
        st.session_state["pending_ddl"] = None
        st.rerun()
    if cancel_col.button("Отменить SQL", use_container_width=True):
        st.session_state["pending_ddl"] = None
        st.session_state["messages"].append(
            {"role": "assistant", "content": "Операция отменена. База данных не изменена."}
        )
        st.rerun()





