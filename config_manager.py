"""
Configuration manager for database and model settings.
Supports runtime configuration through Streamlit UI and omniroute API integration.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger("jarvis.config")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / ".jarvis_config.json"
PLACEHOLDER_API_KEY = "replace_with_a_new_key"
load_dotenv(BASE_DIR / ".env")



def _is_placeholder(value: str | None) -> bool:
    if not isinstance(value, str):
        return True
    normalized = value.strip().casefold()
    return not normalized or normalized.startswith(("replace_with_", "your_"))


def normalize_omniroute_base_url(base_url: str) -> str:
    """Return an OpenAI-compatible API base URL ending in /v1."""
    base_url = base_url.strip().rstrip("/") or "http://localhost:20128"
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str
    port: int
    name: str
    user: str
    password: str

    def to_pymssql_dict(self) -> dict[str, object]:
        """Convert to pymssql connection parameters."""
        return {
            "server": self.host,
            "port": str(self.port),
            "database": self.name,
            "user": self.user,
            "password": self.password,
            "login_timeout": 5,
            "timeout": 15,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding password for logs)."""
        return {
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "user": self.user,
        }


@dataclass
class ModelConfig:
    """Model configuration."""

    provider: str  # "gemini", "omniroute", etc.
    model_name: str
    api_key: str
    base_url: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding API key for logs)."""
        result = {
            "provider": self.provider,
            "model_name": self.model_name,
        }
        if self.base_url:
            result["base_url"] = self.base_url
        return result


@dataclass
class SupersetMCPConfig:
    """Connection settings for an Apache Superset MCP server."""

    enabled: bool = False
    endpoint_url: str = ""
    bearer_token: str = ""


class OmnirouteClient:
    """Client for omniroute API to fetch available models."""

    def __init__(self, api_key: str, base_url: str = "http://localhost:20128"):
        self.api_key = api_key
        self.base_url = normalize_omniroute_base_url(base_url)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def get_available_models(self) -> list[dict]:
        """Fetch list of available models from omniroute API.

        Returns:
            List of model configurations with name, description, and capabilities.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=10,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            logger.info(f"Fetched {len(models)} models from omniroute")
            return models
        except requests.RequestException as e:
            logger.error(f"Failed to fetch models from omniroute: {e}")
            return []

    def test_connection(self) -> bool:
        """Test connection to omniroute API.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Omniroute connection test failed: {e}")
            return False


class ConfigManager:
    """Manages application configuration with persistence."""

    def __init__(self):
        self._db_config: Optional[DatabaseConfig] = None
        self._model_config: Optional[ModelConfig] = None
        self._superset_mcp_config: Optional[SupersetMCPConfig] = None
        self._load_from_file()

    def _load_from_file(self) -> None:
        """Load configuration from JSON file if it exists."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)

                if "database" in data:
                    db = data["database"]
                    self._db_config = DatabaseConfig(
                        host=db["host"],
                        port=db["port"],
                        name=db["name"],
                        user=db["user"],
                        password=db["password"],
                    )

                if "model" in data:
                    m = data["model"]
                    self._model_config = ModelConfig(
                        provider=m["provider"],
                        model_name=m["model_name"],
                        api_key=m["api_key"],
                        base_url=m.get("base_url"),
                    )

                if "superset_mcp" in data:
                    superset = data["superset_mcp"]
                    if not isinstance(superset, dict):
                        raise ValueError("Invalid Superset MCP configuration.")
                    enabled = superset.get("enabled", False)
                    endpoint_url = superset.get("endpoint_url", "")
                    bearer_token = superset.get("bearer_token", "")
                    if (
                        not isinstance(enabled, bool)
                        or not isinstance(endpoint_url, str)
                        or not isinstance(bearer_token, str)
                    ):
                        raise ValueError("Invalid Superset MCP configuration.")
                    self._superset_mcp_config = SupersetMCPConfig(
                        enabled=enabled,
                        endpoint_url=endpoint_url,
                        bearer_token=bearer_token,
                    )

                logger.info("Configuration loaded from file")
            except Exception as e:
                logger.error(f"Failed to load configuration file: {e}")

    def _save_to_file(self) -> None:
        """Save current configuration to JSON file."""
        data = {}

        if self._db_config:
            data["database"] = {
                "host": self._db_config.host,
                "port": self._db_config.port,
                "name": self._db_config.name,
                "user": self._db_config.user,
                "password": self._db_config.password,
            }

        if self._model_config:
            data["model"] = {
                "provider": self._model_config.provider,
                "model_name": self._model_config.model_name,
                "api_key": self._model_config.api_key,
            }
            if self._model_config.base_url:
                data["model"]["base_url"] = self._model_config.base_url

        if self._superset_mcp_config:
            data["superset_mcp"] = {
                "enabled": self._superset_mcp_config.enabled,
                "endpoint_url": self._superset_mcp_config.endpoint_url,
                "bearer_token": self._superset_mcp_config.bearer_token,
            }

        temporary_file = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
        try:
            temporary_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.chmod(temporary_file, 0o600)
            os.replace(temporary_file, CONFIG_FILE)
            logger.info("Configuration saved to file")
        except Exception:
            logger.exception("Failed to save configuration")
            temporary_file.unlink(missing_ok=True)
            raise

    def get_database_config(self) -> Optional[DatabaseConfig]:
        """Get current database configuration."""
        if self._db_config:
            return self._db_config

        # Try to load from environment
        required = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
        if all(not _is_placeholder(os.getenv(name, "")) for name in required):
            try:
                port = int(os.getenv("DB_PORT", "1433"))
            except ValueError as exc:
                raise RuntimeError("DB_PORT должен быть числом.") from exc
            if not 1 <= port <= 65535:
                raise RuntimeError("DB_PORT должен быть в диапазоне от 1 до 65535.")

            self._db_config = DatabaseConfig(
                host=os.getenv("DB_HOST", "").strip(),
                port=port,
                name=os.getenv("DB_NAME", "").strip(),
                user=os.getenv("DB_USER", "").strip(),
                password=os.getenv("DB_PASSWORD", ""),
            )
            return self._db_config

        return None

    def set_database_config(self, config: DatabaseConfig) -> None:
        """Set and persist database configuration."""
        previous_config = self._db_config
        self._db_config = config
        try:
            self._save_to_file()
        except Exception:
            self._db_config = previous_config
            raise
        logger.info(f"Database config updated: {config.to_dict()}")

    def get_model_config(self) -> Optional[ModelConfig]:
        """Get current model configuration.

        Priority order:
        1. File config (.jarvis_config.json) - highest priority when exists
        2. Environment variables (fallback for backward compatibility)
        """
        # First check if we have a saved config in memory
        if self._model_config:
            return self._model_config

        # Try to load from saved file FIRST (user's explicit choice)
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)

                if "model" in data:
                    m = data["model"]
                    provider = m.get("provider", "gemini")
                    model_name = m.get("model_name", "")
                    api_key = m.get("api_key", "")

                    # Skip loading if it's just the default template value
                    if model_name and not _is_placeholder(api_key):
                        self._model_config = ModelConfig(
                            provider=provider,
                            model_name=model_name,
                            api_key=api_key,
                            base_url=m.get("base_url"),
                        )
                        logger.info(f"Loaded model config from file: {self._model_config.to_dict()}")
                        return self._model_config
                    else:
                        logger.warning("File has incomplete model config, checking env...")

            except Exception as e:
                logger.error(f"Failed to load model config from file: {e}")

        # Fallback to environment variables (backward compatibility)
        omni_api_key = os.getenv("OMNIROUTE_API_KEY", "").strip()
        omni_base_url = os.getenv("OMNIROUTE_BASE_URL", "").strip()

        # Check omniroute first (if explicitly set in env)
        if not _is_placeholder(omni_api_key) and not self._model_config:
            self._model_config = ModelConfig(
                provider="omniroute",
                model_name="default",
                api_key=omni_api_key,
                base_url=omni_base_url if omni_base_url else None,
            )
            logger.info("Loaded omniroute config from environment")
            return self._model_config

        # Then check gemini
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = os.getenv("GEMINI_MODEL", "").strip()

        if not _is_placeholder(api_key) and not _is_placeholder(model_name):
            self._model_config = ModelConfig(
                provider="gemini",
                model_name=model_name,
                api_key=api_key,
            )
            logger.info("Loaded Gemini config from environment")
            return self._model_config

        return None

    def set_model_config(self, config: ModelConfig) -> None:
        """Set and persist model configuration."""
        previous_config = self._model_config
        self._model_config = config
        try:
            self._save_to_file()
        except Exception:
            self._model_config = previous_config
            raise
        logger.info(f"Model config updated: {config.to_dict()}")

    def get_superset_mcp_config(self) -> SupersetMCPConfig:
        """Get Superset MCP settings, preferring saved values over environment."""
        if os.getenv("JARVIS_DISABLE_SUPERSET_MCP", "").strip().lower() == "true":
            return SupersetMCPConfig()
        if self._superset_mcp_config:
            return self._superset_mcp_config

        return SupersetMCPConfig(
            enabled=os.getenv("SUPERSET_MCP_ENABLED", "false").strip().lower()
            == "true",
            endpoint_url=os.getenv("SUPERSET_MCP_URL", "").strip(),
            bearer_token=os.getenv("SUPERSET_MCP_BEARER_TOKEN", "").strip(),
        )

    def set_superset_mcp_config(self, config: SupersetMCPConfig) -> None:
        """Set and persist Superset MCP settings."""
        previous_config = self._superset_mcp_config
        self._superset_mcp_config = config
        try:
            self._save_to_file()
        except Exception:
            self._superset_mcp_config = previous_config
            raise

    def get_omniroute_client(self) -> Optional[OmnirouteClient]:
        """Get omniroute client if configured."""
        model_config = self.get_model_config()
        if model_config and model_config.provider == "omniroute" and model_config.api_key:
            return OmnirouteClient(
                model_config.api_key,
                model_config.base_url or "http://localhost:20128",
            )

        api_key = os.getenv("OMNIROUTE_API_KEY", "").strip()
        if _is_placeholder(api_key):
            return None

        base_url = os.getenv(
            "OMNIROUTE_BASE_URL", "http://localhost:20128"
        ).strip()
        return OmnirouteClient(api_key, base_url)

    def is_ui_config_enabled(self) -> bool:
        """Check if UI-based configuration is enabled."""
        return os.getenv("ALLOW_DB_CONFIG_UI", "true").lower() == "true"
