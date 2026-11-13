import asyncio
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path

import pymssql
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

from config_manager import (
    ConfigManager,
    PLACEHOLDER_API_KEY,
    normalize_omniroute_base_url,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("jarvis")

# Initialize configuration manager
config_manager = ConfigManager()

# Get model config
model_config = config_manager.get_model_config()
GEMINI_API_KEY = model_config.api_key if model_config else os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = model_config.model_name if model_config else os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
MAX_MESSAGE_LENGTH = 6000
MAX_QUERY_ROWS = 100
DDL_CONFIRMATION_TTL_SECONDS = 300
MAX_OMNIROUTE_TOOL_ROUNDS = 6
MAX_OMNIROUTE_TOOL_CALLS = 8
MAX_OMNIROUTE_HISTORY_TURNS = 10
MAX_GEMINI_TOOL_ROUNDS = 25

SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", re.IGNORECASE)
SQL_ASSISTANT_INSTRUCTION = (
    "Ты — J.A.R.V.I.S., SQL-аналитик Microsoft SQL Server. Всегда отвечай "
    "по-русски и используй Markdown. Для вопросов о данных или структуре БД "
    "обязательно используй инструмент execute_read_only_query; не утверждай, "
    "что у тебя нет доступа к базе, если инструмент ещё не вызван. При "
    "необходимости сначала изучи схему через INFORMATION_SCHEMA или sys-таблицы. "
    "Выполняй только одиночные T-SQL SELECT-запросы. Для создания объектов "
    "предложи один запрос CREATE TABLE, CREATE VIEW или CREATE INDEX в отдельном "
    "блоке SQL; изменения разрешены только после подтверждения пользователем. "
    "Не предлагай DROP, ALTER, UPDATE, DELETE, INSERT или несколько команд. "
    "В итоговом ответе укажи, какие запросы выполнил, и честно сообщи, если "
    "инструмент вернул ошибку."
)
ALLOWED_DDL = re.compile(
    r"^\s*CREATE\s+(?:TABLE|VIEW|(?:UNIQUE\s+)?"
    r"(?:(?:CLUSTERED|NONCLUSTERED)\s+)?INDEX)\b",
    re.IGNORECASE,
)
READ_ONLY_BLOCKED_WORDS = {
    "ALTER",
    "BACKUP",
    "BULK",
    "CREATE",
    "DBCC",
    "DENY",
    "DELETE",
    "DROP",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "INSERT",
    "INTO",
    "MERGE",
    "RESTORE",
    "REVOKE",
    "TRUNCATE",
    "UPDATE",
    "USE",
    "WAITFOR",
}
OMNIROUTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_read_only_query",
            "description": (
                "Выполнить один безопасный T-SQL SELECT к настроенной базе "
                "Microsoft SQL Server. Результат ограничен 100 строками."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "Один T-SQL SELECT-запрос.",
                    }
                },
                "required": ["sql_query"],
                "additionalProperties": False,
            },
        },
    }
]

app = FastAPI(title="J.A.R.V.I.S. SQL Assistant")


def database_config() -> dict[str, object]:
    """Build MS SQL Server connection settings from ConfigManager for pymssql."""
    db_config = config_manager.get_database_config()
    if not db_config:
        raise RuntimeError(
            "Не заполнены настройки базы данных. "
            "Заполните конфигурацию через интерфейс или установите переменные окружения."
        )

    return db_config.to_pymssql_dict()


def _tokenize_sql(sql: str) -> list[tuple[str, str]] | None:
    """Tokenize enough T-SQL to reject comments, batches, and write statements."""
    tokens: list[tuple[str, str]] = []
    index = 0

    while index < len(sql):
        char = sql[index]
        if char.isspace():
            index += 1
            continue
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline == -1 else newline + 1
            continue
        if sql.startswith("/*", index):
            depth = 1
            index += 2
            while index < len(sql) and depth:
                if sql.startswith("/*", index):
                    depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                return None
            continue
        if char == "'":
            index += 1
            while index < len(sql):
                if sql[index] == "'":
                    if index + 1 < len(sql) and sql[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                return None
            tokens.append(("literal", ""))
            continue
        if char in ('[', '"'):
            closing = "]" if char == "[" else '"'
            index += 1
            while index < len(sql):
                if sql[index] == closing:
                    if index + 1 < len(sql) and sql[index + 1] == closing:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                return None
            tokens.append(("identifier", ""))
            continue
        if char.isalpha() or char in "_@#":
            start = index
            index += 1
            while index < len(sql) and (
                sql[index].isalnum() or sql[index] in "_$@#"
            ):
                index += 1
            tokens.append(("word", sql[start:index].upper()))
            continue
        tokens.append(("symbol", char))
        index += 1

    return tokens


def _matching_parenthesis(tokens: list[tuple[str, str]], start: int) -> int | None:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index] == ("symbol", "("):
            depth += 1
        elif tokens[index] == ("symbol", ")"):
            depth -= 1
            if depth == 0:
                return index
    return None


def is_read_only_select(sql_query: str) -> bool:
    """Accept one SELECT statement, optionally preceded by read-only CTEs."""
    tokens = _tokenize_sql(sql_query)
    if not tokens:
        return False

    semicolons = [index for index, token in enumerate(tokens) if token == ("symbol", ";")]
    if semicolons:
        if len(semicolons) != 1 or semicolons[0] != len(tokens) - 1:
            return False
        tokens.pop()
    if not tokens:
        return False

    main_statement = 0
    if tokens[0] == ("word", "WITH"):
        index = 1
        while True:
            if index >= len(tokens) or tokens[index][0] not in {"word", "identifier"}:
                return False
            index += 1
            if index < len(tokens) and tokens[index] == ("symbol", "("):
                closing = _matching_parenthesis(tokens, index)
                if closing is None:
                    return False
                index = closing + 1
            if index >= len(tokens) or tokens[index] != ("word", "AS"):
                return False
            index += 1
            if index >= len(tokens) or tokens[index] != ("symbol", "("):
                return False
            closing = _matching_parenthesis(tokens, index)
            if closing is None:
                return False
            index = closing + 1
            if index < len(tokens) and tokens[index] == ("symbol", ","):
                index += 1
                continue
            main_statement = index
            break

    if main_statement >= len(tokens) or tokens[main_statement] != ("word", "SELECT"):
        return False
    return not any(
        kind == "word" and value in READ_ONLY_BLOCKED_WORDS
        for kind, value in tokens[main_statement:]
    )


def execute_read_only_query(sql_query: str) -> str:
    """Execute a bounded read-only MS SQL Server query for Gemini."""
    query = sql_query.strip()
    if not is_read_only_select(query):
        return "Запрос отклонён: разрешены только SELECT-запросы."

    connection = None
    try:
        connection = pymssql.connect(**database_config())
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [description[0] for description in cursor.description or ()]
            rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)
            truncated = len(rows) > MAX_QUERY_ROWS

            # Convert rows to list of dicts
            data = []
            for row in rows[:MAX_QUERY_ROWS]:
                data.append(dict(zip(columns, row)))

            result = {
                "columns": columns,
                "data": data,
                "truncated": truncated,
            }
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        logger.exception("Read-only database query failed")
        return "Не удалось выполнить запрос к базе данных. Проверьте подключение и SQL."
    finally:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()


def is_approved_ddl(sql_statement: str) -> bool:
    """Allow only one CREATE TABLE, VIEW, or INDEX statement."""
    statement = sql_statement.strip()
    if not ALLOWED_DDL.match(statement):
        return False
    tokens = _tokenize_sql(statement)
    if not tokens:
        return False
    semicolons = [index for index, token in enumerate(tokens) if token == ("symbol", ";")]
    return not semicolons or (
        len(semicolons) == 1 and semicolons[0] == len(tokens) - 1
    )


def execute_ddl_statement(sql_statement: str) -> str:
    if not is_approved_ddl(sql_statement):
        return "Запрос отклонён: разрешены только одиночные CREATE TABLE, VIEW или INDEX."

    connection = None
    try:
        connection = pymssql.connect(**database_config())
        with connection.cursor() as cursor:
            cursor.execute(sql_statement)
        connection.commit()
        return "SQL выполнен успешно."
    except Exception:
        logger.exception("Confirmed DDL statement failed")
        return "Не удалось выполнить SQL. Проверьте подключение, права пользователя и запрос."
    finally:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()


class ChatResponse:
    def __init__(self, text: str):
        self.text = text


class GeminiResponseError(RuntimeError):
    pass


class GeminiChat:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            tools=[execute_read_only_query],
            system_instruction=SQL_ASSISTANT_INSTRUCTION,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_GEMINI_TOOL_ROUNDS,
            ),
        )
        self.chat = self.client.chats.create(model=model_name, config=self.config)

    def _restore_history(self, history: list[types.Content]) -> None:
        self.chat = self.client.chats.create(
            model=self.model_name, config=self.config, history=history
        )

    @staticmethod
    def _empty_response_error(response: types.GenerateContentResponse) -> GeminiResponseError:
        details = []
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            details.append(f"блокировка: {response.prompt_feedback.block_reason.value}")
        if response.candidates and response.candidates[0].finish_reason:
            details.append(f"завершение: {response.candidates[0].finish_reason.value}")
        if response.function_calls:
            details.append("модель вернула вызов инструмента без итогового ответа")
        reason = "; ".join(details) if details else "причина не указана API"
        logger.warning("Gemini returned no text: %s", reason)
        return GeminiResponseError(f"Gemini не вернул текстовый ответ ({reason}).")

    def send_message(self, prompt: str) -> types.GenerateContentResponse | ChatResponse:
        previous_history = self.chat.get_history(curated=True)
        response = self.chat.send_message(prompt)
        history = response.automatic_function_calling_history
        if (
            response.function_calls
            and history
            and history[-1].parts
            and any(part.function_response for part in history[-1].parts)
        ):
            try:
                summary_config = self.config.model_copy(update={
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    "tool_config": types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="NONE")
                    ),
                    "system_instruction": (
                        SQL_ASSISTANT_INSTRUCTION
                        + " Лимит раундов обращения к инструментам достигнут. "
                        "Не вызывай инструменты; "
                        "подведи итоги только по уже полученным данным и укажи, "
                        "какие части задачи остались непроверенными."
                    ),
                })
                summary = self.client.models.generate_content(
                    model=self.model_name, contents=history, config=summary_config
                )
                if not summary.text or summary.function_calls:
                    raise self._empty_response_error(summary)
                if not summary.candidates or not summary.candidates[0].content:
                    raise self._empty_response_error(summary)
                self._restore_history([*history, summary.candidates[0].content])
                return ChatResponse(
                    "⚠️ Достигнут лимит раундов обращения к базе; результаты могут быть неполными.\n\n"
                    + summary.text
                )
            except Exception:
                self._restore_history(previous_history)
                raise

        if not response.text or response.function_calls:
            self._restore_history(previous_history)
            raise self._empty_response_error(response)
        return response

    def close(self) -> None:
        self.client.close()


class OmnirouteChat:
    """OpenAI-compatible chat client exposing only the bounded SQL read tool."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.history: list[dict[str, object]] = []

    def send_message(self, prompt: str) -> ChatResponse:
        conversation = [*self.history, {"role": "user", "content": prompt}]
        tool_call_count = 0

        for _ in range(MAX_OMNIROUTE_TOOL_ROUNDS):
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": SQL_ASSISTANT_INSTRUCTION},
                    *conversation,
                ],
                "temperature": 0.1,
                "parallel_tool_calls": False,
                "stream": False,
                "tools": OMNIROUTE_TOOLS,
                "tool_choice": "auto",
            }
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                logger.exception("Omniroute API request failed")
                raise RuntimeError(f"Ошибка при обращении к Omniroute: {exc}") from exc

            choices = result.get("choices") if isinstance(result, dict) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError("Omniroute вернул ответ без choices.")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise RuntimeError("Omniroute вернул некорректное сообщение.")

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Omniroute вернул пустой ответ.")
                conversation.append({"role": "assistant", "content": content})
                user_turns = [
                    index
                    for index, item in enumerate(conversation)
                    if item.get("role") == "user"
                ]
                if len(user_turns) > MAX_OMNIROUTE_HISTORY_TURNS:
                    conversation = conversation[user_turns[-MAX_OMNIROUTE_HISTORY_TURNS] :]
                self.history = conversation
                return ChatResponse(content)

            if not isinstance(tool_calls, list):
                raise RuntimeError("Omniroute вернул некорректный список инструментов.")
            if tool_call_count + len(tool_calls) > MAX_OMNIROUTE_TOOL_CALLS:
                raise RuntimeError("Превышен лимит SQL-запросов за одно сообщение.")

            conversation.append(message)
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    raise RuntimeError("Omniroute вернул некорректный вызов инструмента.")
                call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(call_id, str) or not isinstance(function, dict):
                    raise RuntimeError("Omniroute вернул некорректный вызов инструмента.")
                if function.get("name") != "execute_read_only_query":
                    raise RuntimeError("Omniroute запросил неизвестный инструмент.")

                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("Omniroute передал некорректные аргументы SQL.") from exc
                if not isinstance(arguments, dict):
                    raise RuntimeError("Omniroute передал некорректные аргументы SQL.")
                sql_query = arguments.get("sql_query")
                if not isinstance(sql_query, str):
                    raise RuntimeError("Omniroute не передал SQL-запрос.")

                tool_result = execute_read_only_query(sql_query)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_result,
                    }
                )
                tool_call_count += 1

        raise RuntimeError("Модель превысила лимит шагов обращения к базе.")


def create_chat():
    """Create chat with selected model (Gemini or omniroute)."""
    model_config = config_manager.get_model_config()

    logger.info(f"Creating chat with model config: {model_config.to_dict() if model_config else 'None'}")

    if not model_config:
        raise RuntimeError(
            "Модель не настроена. Заполните конфигурацию в боковой панели."
        )

    if model_config.provider == "omniroute":
        if not model_config.api_key or model_config.api_key == PLACEHOLDER_API_KEY:
            raise RuntimeError("OMNIROUTE_API_KEY не установлен.")

        base_url = normalize_omniroute_base_url(
            model_config.base_url or "http://localhost:20128"
        )
        return OmnirouteChat(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=base_url,
        )

    if model_config.provider != "gemini":
        raise RuntimeError(f"Неизвестный провайдер модели: {model_config.provider}")

    else:  # gemini provider
        if not model_config.api_key or model_config.api_key == PLACEHOLDER_API_KEY:
            raise RuntimeError("Укажите действительный Gemini API Key в конфигурации.")

        return GeminiChat(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
        )


def test_model_connection() -> dict[str, bool | str]:
    """Test if the selected model is working with a simple prompt."""
    try:
        model_config = config_manager.get_model_config()

        if not model_config:
            return {"success": False, "error": "Модель не настроена"}

        if model_config.provider not in {"omniroute", "gemini"}:
            return {
                "success": False,
                "error": f"Неизвестный провайдер модели: {model_config.provider}",
            }

        if model_config.provider == "omniroute":
            import requests

            if not model_config.api_key:
                return {"success": False, "error": "OMNIROUTE_API_KEY не установлен"}

            session = requests.Session()
            session.headers.update({
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json",
            })

            base_url = normalize_omniroute_base_url(
                model_config.base_url or "http://localhost:20128"
            )
            payload = {
                "model": model_config.model_name,
                "messages": [
                    {"role": "user", "content": "1+1=? ответь только числом!"}
                ],
                "temperature": 0.1,
            }

            try:
                response = session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not isinstance(answer, str) or not answer.strip():
                    return {"success": False, "error": "Модель вернула пустой ответ"}
                return {"success": True, "answer": answer.strip()}

            except requests.RequestException as e:
                return {"success": False, "error": f"Ошибка omniroute: {str(e)}"}

        else:  # gemini
            if not model_config.api_key or model_config.api_key == PLACEHOLDER_API_KEY:
                return {"success": False, "error": "Gemini API Key не установлен"}

            client = genai.Client(api_key=model_config.api_key)
            response = client.models.generate_content(
                model=model_config.model_name,
                contents="1+1=? ответь только числом!",
            )

            if response.text and response.text.strip():
                return {"success": True, "answer": response.text.strip()}
            else:
                return {
                    "success": False,
                    "error": str(GeminiChat._empty_response_error(response)),
                }

    except Exception as e:
        logger.exception("Model connection test failed")
        return {"success": False, "error": f"Ошибка: {str(e)}"}


@app.get("/")
async def get_homepage():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health():
    missing = []
    if config_manager.get_database_config() is None:
        missing.extend(("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"))
    if config_manager.get_model_config() is None:
        missing.append("GEMINI_API_KEY or OMNIROUTE_API_KEY")
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

                pending_ddl.clear()
                try:
                    response = await asyncio.to_thread(chat.send_message, user_text.strip())
                    answer = response.text
                    if not answer or not answer.strip():
                        raise RuntimeError("Модель не вернула текстовый ответ.")
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
                except GeminiResponseError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    logger.exception("Gemini request failed")
                    model_config = config_manager.get_model_config()
                    provider_name = (
                        model_config.provider
                        if model_config
                        else "модели"
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": (
                                f"Не удалось получить ответ от {provider_name}. "
                                "Проверьте ключ, модель и интернет-соединение."
                            ),
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
    finally:
        if isinstance(chat, GeminiChat):
            chat.close()





