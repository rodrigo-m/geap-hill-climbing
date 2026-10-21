"""Configuration settings for the Sig Normalizer ADK agent and GEAP tooling."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from package directory or repository root if present
for _candidate_env in (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
):
    if _candidate_env.exists():
        load_dotenv(_candidate_env, override=False)
        break
else:
    load_dotenv(override=False)

# Enforce global defaults for Gemini 3.8 Flash on Vertex AI and enable telemetry/logging
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("LLM_LOCATION", "global")
os.environ.setdefault("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
os.environ.setdefault("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true")
os.environ.setdefault("ENABLE_PROMPT_RESPONSE_LOGGING", "true")


@dataclass(frozen=True)
class Settings:
    project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    # Gemini 3.8 Flash is served from the 'global' multiregion endpoint.
    model: str = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
    llm_location: str = os.environ.get("LLM_LOCATION", "global")
    # Vertex AI Prompt Management & Evaluation resources live in a regional endpoint.
    resource_location: str = os.environ.get("RESOURCE_LOCATION", "us-central1")
    prompt_env: str = os.environ.get("PROMPT_ENV", "staging")
    prompt_cache_ttl_seconds: int = int(os.environ.get("PROMPT_CACHE_TTL_SECONDS", "60"))
    release_bucket: str = os.environ.get("PROMPT_RELEASE_BUCKET", "")
    enable_telemetry: bool = (
        os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "true").lower()
        in ("true", "1", "yes")
    )
    capture_message_content: bool = (
        os.environ.get(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true"
        ).lower()
        in ("true", "1", "yes", "event_only", "span_only", "span_and_event")
    )
    capture_message_content_in_spans: bool = (
        os.environ.get("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true").lower()
        in ("true", "1", "yes")
    )
    enable_prompt_response_logging: bool = (
        os.environ.get("ENABLE_PROMPT_RESPONSE_LOGGING", "true").lower()
        in ("true", "1", "yes")
    )


settings = Settings()
