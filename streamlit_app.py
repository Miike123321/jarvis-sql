import logging
import os
import re

import pymssql
import streamlit as st

logger = logging.getLogger("jarvis")

from config_manager import DatabaseConfig, ModelConfig, PLACEHOLDER_API_KEY
from jarvis_backend import (
    GEMINI_MODEL,
    GeminiChat,
    GeminiResponseError,
    MAX_MESSAGE_LENGTH,
    create_chat,
    database_config,
    execute_ddl_statement,
    is_approved_ddl,
    config_manager,
    test_model_connection,
)


SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", re.IGNORECASE)
REQUIRED_SETTINGS = (
    "GEMINI_API_KEY",
    "DB_HOST",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
)

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
            gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if current_model and current_model.provider == "gemini":
                st.success(f"✅ Используется: **{current_model.model_name}**")
            elif gemini_api_key and gemini_api_key != PLACEHOLDER_API_KEY:
                new_model_config = ModelConfig(
                    provider="gemini",
                    model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
                    api_key=gemini_api_key,
                )
                try:
                    config_manager.set_model_config(new_model_config)
                    reset_model_session()
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Не удалось сохранить модель: {exc}")
            else:
                st.warning("⚠️ Настройте GEMINI_API_KEY в файле .env.")


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
            st.session_state["chat"] = create_chat()
            st.session_state["_last_chat_provider"] = current_provider
        else:
            st.session_state["chat"] = create_chat()
    except Exception as exc:
        st.session_state["chat_error"] = str(exc)

if st.session_state["chat_error"]:
    st.error(st.session_state["chat_error"])

for saved_message in st.session_state["messages"]:
    show_message(saved_message)

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "Здравствуйте! Добавьте одну или несколько частей задачи, "
            "затем отправьте их одним запросом."
        )

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





