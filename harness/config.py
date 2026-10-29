"""Persistent configuration: API keys, model, working folder, agent limits.

Stored as JSON at ``$HARNESS_HOME/config.json`` (default ``~/.harness``).
API keys are stored by environment-variable name (e.g. ``ANTHROPIC_API_KEY``)
and exported into ``os.environ`` at startup, which is where litellm reads them.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_MODEL = "anthropic/claude-sonnet-5-5"

# Shorthand provider names -> the env var litellm expects.
PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "together": "TOGETHERAI_API_KEY",
    "cohere": "COHERE_API_KEY",
}

# Fields settable via `config set` / `/set`, with their coercion.
SETTABLE = {
    "model": str,
    "workdir": str,
    "api_base": str,
    "max_steps": int,
    "temperature": float,
    "system_prompt": str,
}


def config_dir() -> Path:
    return Path(os.environ.get("HARNESS_HOME") or Path.home() / ".harness")


def config_path() -> Path:
    return config_dir() / "config.json"


def key_env_name(name: str) -> str:
    """Map 'anthropic' -> 'ANTHROPIC_API_KEY'; pass through explicit env var names."""
    return PROVIDER_ENV.get(name.lower(), name.upper())


def missing_keys(model: str, api_base: str | None = None) -> list[str]:
    """Best-effort check that the model's provider has a key, without importing litellm."""
    if api_base:
        return []
    provider = model.split("/", 1)[0].lower() if "/" in model else ""
    if not provider:
        name = model.lower()
        provider = next((p for hint, p in (("claude", "anthropic"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
                                            ("o4", "openai"), ("gemini", "gemini")) if name.startswith(hint)), "")
    env = PROVIDER_ENV.get(provider)
    return [env] if env and not os.environ.get(env) else []


def mask(secret: str) -> str:
    if len(secret) <= 10:
        return "•" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    workdir: str | None = None  # None -> the folder harness was launched from
    api_base: str | None = None  # custom endpoint, e.g. http://localhost:11434 for ollama
    max_steps: int = 30
    temperature: float | None = None
    system_prompt: str | None = None  # extra instructions appended to the built-in prompt
    api_keys: dict[str, str] = field(default_factory=dict)
    auto_approve: list[str] = field(default_factory=list)  # tool names that skip confirmation

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls) -> "Config":
        path = config_path()
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> Path:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path

    # -- mutation ----------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        if name not in SETTABLE:
            raise KeyError(f"unknown setting '{name}' (settable: {', '.join(SETTABLE)})")
        if value.lower() in ("", "none", "null"):
            if name in ("model", "max_steps"):
                raise ValueError(f"'{name}' cannot be cleared")
            setattr(self, name, None)
            return
        if name == "workdir":
            path = Path(value).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"not a directory: {path}")
            value = str(path)
        try:
            setattr(self, name, SETTABLE[name](value))
        except ValueError:
            raise ValueError(f"'{name}' must be a{'n integer' if SETTABLE[name] is int else ' number'}, got '{value}'")

    def set_key(self, name: str, secret: str) -> str:
        env = key_env_name(name)
        self.api_keys[env] = secret.strip()
        return env

    def remove_key(self, name: str) -> str:
        env = key_env_name(name)
        if env not in self.api_keys:
            raise KeyError(f"no key stored for {env}")
        del self.api_keys[env]
        return env

    # -- runtime -----------------------------------------------------------

    def apply_env(self) -> None:
        """Export stored keys so litellm picks them up. Stored keys win over the shell's."""
        os.environ.update(self.api_keys)

    def resolved_workdir(self) -> Path:
        return Path(self.workdir).expanduser().resolve() if self.workdir else Path.cwd().resolve()

    def display_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("model", self.model),
            ("workdir", str(self.resolved_workdir()) + ("" if self.workdir else "  (launch folder)")),
            ("api_base", self.api_base or "—"),
            ("max_steps", str(self.max_steps)),
            ("temperature", "default" if self.temperature is None else str(self.temperature)),
            ("system_prompt", (self.system_prompt[:60] + "…") if self.system_prompt and len(self.system_prompt) > 60 else (self.system_prompt or "—")),
            ("auto_approve", ", ".join(self.auto_approve) or "—"),
        ]
        for env, secret in sorted(self.api_keys.items()):
            rows.append((env, mask(secret)))
        if not self.api_keys:
            rows.append(("api keys", "none — set one with `key <provider> <key>`"))
        rows.append(("config file", str(config_path())))
        return rows
