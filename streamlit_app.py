import os
import re

import psycopg2
import streamlit as st

from jarvis_backend import (
    GEMINI_MODEL,
    MAX_MESSAGE_LENGTH,
    create_chat,
    database_config,
    execute_ddl_statement,
    is_approved_ddl,
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


initialize_session()

st.title("⚡ J.A.R.V.I.S.")
st.caption("SQL-ассистент · Gemini + PostgreSQL")

with st.sidebar:
    st.subheader("Состояние")
    st.caption(f"Модель: `{GEMINI_MODEL}`")
    missing_settings = [
        name for name in REQUIRED_SETTINGS if not os.getenv(name, "").strip()
    ]
    if missing_settings:
        st.error("Нужно заполнить настройки в `.env`.")
        st.code("\n".join(missing_settings), language="text")
    else:
        st.success("Настройки окружения заполнены")

    if st.button("Проверить подключение к PostgreSQL", use_container_width=True):
        try:
            with st.spinner("Проверяю подключение..."):
                with psycopg2.connect(**database_config()) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
            st.success("Подключение к PostgreSQL работает")
        except Exception:
            st.error(
                "Не удалось подключиться. Проверьте DB_HOST, порт, доступ к сети "
                "и учётные данные."
            )

    st.divider()
    st.caption(
        "Сообщения копятся в черновике. Ассистент получит их только после нажатия "
        "«Отправить ассистенту»."
    )

if st.session_state["chat"] is None and not missing_settings:
    try:
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

    send_disabled = (
        st.session_state["chat"] is None
        or bool(missing_settings)
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
                with st.spinner("J.A.R.V.I.S. анализирует задачу..."):
                    response = st.session_state["chat"].send_message(combined_prompt)
                answer = response.text or "Модель вернула пустой ответ."
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
            except Exception:
                st.error(
                    "Не удалось получить ответ от Gemini. Проверьте API-ключ, "
                    "название модели и интернет-соединение."
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
