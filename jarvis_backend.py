import asyncio
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from psycopg2.extras import RealDictCursor


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("jarvis")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
MAX_MESSAGE_LENGTH = 6000
MAX_QUERY_ROWS = 100
QUERY_TIMEOUT_MS = 15_000
DDL_CONFIRMATION_TTL_SECONDS = 300

SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", re.IGNORECASE)
ALLOWED_DDL = re.compile(
    r"^\s*CREATE\s+(?:(?:OR\s+REPLACE|UNIQUE)\s+)?"
    r"(?:TABLE|VIEW|INDEX)\b",
    re.IGNORECASE,
)

app = FastAPI(title="J.A.R.V.I.S. SQL Assistant")


def database_config() -> dict[str, object]:
    """Build PostgreSQL connection settings from the environment."""
    required = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Не заполнены настройки базы данных: " + ", ".join(missing)
        )

    try:
        port = int(os.getenv("DB_PORT", "5432"))
    except ValueError as exc:
        raise RuntimeError("DB_PORT должен быть числом.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("DB_PORT должен быть в диапазоне от 1 до 65535.")

    return {
        "host": os.environ["DB_HOST"].strip(),
        "port": port,
        "dbname": os.environ["DB_NAME"].strip(),
        "user": os.environ["DB_USER"].strip(),
        "password": os.environ["DB_PASSWORD"],
        "connect_timeout": 5,
        "application_name": "jarvis-assistant",
    }


def execute_read_only_query(sql_query: str) -> str:
    """Execute a bounded read-only PostgreSQL query for Gemini."""
    query = sql_query.strip()
    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        return "Запрос отклонён: разрешены только SELECT-запросы."

    connection = None
    try:
        connection = psycopg2.connect(**database_config())
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"SET LOCAL statement_timeout = {QUERY_TIMEOUT_MS}")
            cursor.execute(query)
            columns = [description.name for description in cursor.description or ()]
            rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
            truncated = len(rows) > MAX_QUERY_ROWS
            result = {
                "columns": columns,
                "data": rows[:MAX_QUERY_ROWS],
                "truncated": truncated,
            }
        connection.rollback()
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        logger.exception("Read-only database query failed")
        return "Не удалось выполнить запрос к базе данных. Проверьте подключение и SQL."
    finally:
        if connection is not None:
            connection.close()


def is_approved_ddl(sql_statement: str) -> bool:
    """Allow only one CREATE TABLE, VIEW, or INDEX statement."""
    statement = sql_statement.strip()
    if not ALLOWED_DDL.match(statement):
        return False
    statement_without_trailing_semicolon = statement.removesuffix(";").rstrip()
    return ";" not in statement_without_trailing_semicolon


def execute_ddl_statement(sql_statement: str) -> str:
    if not is_approved_ddl(sql_statement):
        return "Запрос отклонён: разрешены только одиночные CREATE TABLE, VIEW или INDEX."

    connection = None
    try:
        connection = psycopg2.connect(**database_config())
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SET LOCAL statement_timeout = {QUERY_TIMEOUT_MS}")
                cursor.execute(sql_statement)
        return "SQL выполнен успешно."
    except Exception:
        logger.exception("Confirmed DDL statement failed")
        return "Не удалось выполнить SQL. Проверьте подключение, права пользователя и запрос."
    finally:
        if connection is not None:
            connection.close()


def create_chat():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "replace_with_a_new_key":
        raise RuntimeError("Укажите действительный GEMINI_API_KEY в файле .env.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    return client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            tools=[execute_read_only_query],
            system_instruction=(
                "Ты — J.A.R.V.I.S., SQL-аналитик PostgreSQL. Отвечай на русском "
                "языке и используй Markdown. Для анализа данных выполняй только "
                "SELECT через инструмент. Не выполняй изменения через инструмент. "
                "Если пользователь явно просит создать объект, предложи один "
                "запрос CREATE TABLE, CREATE VIEW или CREATE INDEX в блоке ```sql. "
                "Не предлагай DROP, DELETE, UPDATE, INSERT, несколько команд или "
                "изменения схемы, кроме разрешённых CREATE-команд."
            ),
            temperature=0.1,
        ),
    )


@app.get("/")
async def get_homepage():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health():
    missing = [
        name
        for name in ("GEMINI_API_KEY", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
        if not os.getenv(name, "").strip()
    ]
    return {
        "status": "needs_configuration" if missing else "configured",
        "missing_settings": missing,
        "model": GEMINI_MODEL,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pending_ddl: dict[str, tuple[str, float]] = {}

    try:
        chat = create_chat()
    except Exception as exc:
        logger.exception("Could not initialize Gemini chat")
        await websocket.send_json(
            {"type": "error", "message": str(exc) or "Не удалось запустить ассистента."}
        )
        await websocket.close(code=1011)
        return

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Получено некорректное сообщение."}
                )
                continue

            if not isinstance(payload, dict):
                await websocket.send_json(
                    {"type": "error", "message": "Неверный формат сообщения."}
                )
                continue

            message_type = payload.get("type")
            if message_type == "user_message":
                user_text = payload.get("text")
                if not isinstance(user_text, str) or not user_text.strip():
                    await websocket.send_json(
                        {"type": "error", "message": "Введите текст сообщения."}
                    )
                    continue
                if len(user_text) > MAX_MESSAGE_LENGTH:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Сообщение слишком длинное (лимит {MAX_MESSAGE_LENGTH} символов).",
                        }
                    )
                    continue

                try:
                    response = await asyncio.to_thread(chat.send_message, user_text.strip())
                    answer = response.text or "Модель вернула пустой ответ."
                    response_payload: dict[str, object] = {
                        "type": "bot_response",
                        "text": answer,
                    }
                    sql_match = SQL_FENCE.search(answer)
                    if sql_match:
                        proposed_sql = sql_match.group(1).strip()
                        if is_approved_ddl(proposed_sql):
                            token = secrets.token_urlsafe(24)
                            pending_ddl[token] = (
                                proposed_sql,
                                time.monotonic() + DDL_CONFIRMATION_TTL_SECONDS,
                            )
                            response_payload["ddl_token"] = token
                        else:
                            response_payload["text"] += (
                                "\n\nЭтот SQL не подходит для безопасного подтверждения: "
                                "разрешены только одиночные CREATE TABLE, VIEW или INDEX."
                            )
                    await websocket.send_json(response_payload)
                except Exception:
                    logger.exception("Gemini request failed")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Не удалось получить ответ от Gemini. Проверьте ключ, модель и интернет-соединение.",
                        }
                    )

            elif message_type == "confirm_ddl":
                token = payload.get("token")
                approved = payload.get("approved")
                if not isinstance(token, str) or not isinstance(approved, bool):
                    await websocket.send_json(
                        {"type": "error", "message": "Некорректное подтверждение SQL."}
                    )
                    continue

                pending = pending_ddl.pop(token, None)
                if pending is None or pending[1] < time.monotonic():
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Подтверждение устарело или уже использовано. Запросите SQL заново.",
                        }
                    )
                    continue

                if approved:
                    result = await asyncio.to_thread(execute_ddl_statement, pending[0])
                    response_text = f"⚙️ **Системный отчёт по SQL:**\n{result}"
                else:
                    response_text = "❌ **Операция отменена.** База данных не изменена."
                await websocket.send_json({"type": "bot_response", "text": response_text})

            else:
                await websocket.send_json(
                    {"type": "error", "message": "Неизвестный тип сообщения."}
                )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")





