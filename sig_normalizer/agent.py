"""Prescription Sig Normalizer ADK Agent."""

from datetime import datetime, timezone
import json
import os
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from opentelemetry import trace

try:
    from .config import settings
    from .prompt_registry import get_prompt_text
except ImportError:
    from config import settings
    from prompt_registry import get_prompt_text


def _extract_text_from_content(content: Any) -> str:
    """Extract concatenated non-thought text from a google.genai Content object."""
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts).strip()


def _extract_system_instruction(llm_request: LlmRequest) -> str:
    """Extract the system instruction string from an LlmRequest."""
    cfg = getattr(llm_request, "config", None)
    sys_inst = getattr(cfg, "system_instruction", None) if cfg else None
    if not sys_inst:
        return ""
    if isinstance(sys_inst, str):
        return sys_inst
    text = _extract_text_from_content(sys_inst)
    return text if text else str(sys_inst)


def _emit_structured_log(payload: dict[str, Any]) -> None:
    """Emit a single-line JSON log record correlated with the active Cloud Trace span."""
    try:
        span = trace.get_current_span()
        span_ctx = span.get_span_context() if span else None
        if span_ctx and getattr(span_ctx, "is_valid", False):
            if settings.project_id:
                payload["logging.googleapis.com/trace"] = (
                    f"projects/{settings.project_id}/traces/{span_ctx.trace_id:032x}"
                )
            payload["logging.googleapis.com/spanId"] = f"{span_ctx.span_id:016x}"
            payload["logging.googleapis.com/trace_sampled"] = bool(
                span_ctx.trace_flags & 0x01
            )
    except Exception:
        pass
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def log_prompt_request(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Enrich the active OpenTelemetry span and log the outgoing LLM prompt request."""
    try:
        state = getattr(callback_context, "state", None) or {}
        env = (
            state.get("prompt_env")
            or os.environ.get("PROMPT_ENV")
            or settings.prompt_env
        )
        pin = state.get("prompt_version_pin") or os.environ.get("PROMPT_VERSION_PIN")

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.prompt.name", "sig-normalizer")
            span.set_attribute("gen_ai.prompt.env", str(env))
            if pin:
                span.set_attribute("gen_ai.prompt.version_pin", str(pin))

        if not settings.enable_prompt_response_logging:
            return None

        user_prompt = _extract_text_from_content(
            getattr(callback_context, "user_content", None)
        )
        if not user_prompt:
            for item in reversed(getattr(llm_request, "contents", None) or []):
                if getattr(item, "role", None) == "user":
                    user_prompt = _extract_text_from_content(item)
                    if user_prompt:
                        break

        sys_inst = _extract_system_instruction(llm_request)
        session = getattr(callback_context, "session", None)
        _emit_structured_log(
            {
                "severity": "INFO",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "gen_ai.prompt_request",
                "agent_name": getattr(callback_context, "agent_name", "sig_normalizer"),
                "invocation_id": getattr(callback_context, "invocation_id", None),
                "session_id": getattr(session, "id", None),
                "user_id": getattr(callback_context, "user_id", None),
                "model": getattr(llm_request, "model", None) or settings.model,
                "prompt_name": "sig-normalizer",
                "prompt_env": env,
                "prompt_version_pin": pin,
                "user_prompt": user_prompt,
                "system_instruction_length": len(sys_inst),
                "system_instruction_preview": sys_inst[:240],
            }
        )
    except Exception:
        pass
    return None


def log_prompt_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Log the completed LLM response paired with the originating prompt and token usage."""
    try:
        if not settings.enable_prompt_response_logging:
            return None
        if getattr(llm_response, "partial", False):
            return None

        state = getattr(callback_context, "state", None) or {}
        env = (
            state.get("prompt_env")
            or os.environ.get("PROMPT_ENV")
            or settings.prompt_env
        )
        pin = state.get("prompt_version_pin") or os.environ.get("PROMPT_VERSION_PIN")

        user_prompt = _extract_text_from_content(
            getattr(callback_context, "user_content", None)
        )
        model_response = _extract_text_from_content(
            getattr(llm_response, "content", None)
        )

        usage = getattr(llm_response, "usage_metadata", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "prompt_token_count": getattr(usage, "prompt_token_count", None),
                "candidates_token_count": getattr(
                    usage, "candidates_token_count", None
                ),
                "total_token_count": getattr(usage, "total_token_count", None),
            }

        finish_reason = getattr(llm_response, "finish_reason", None)
        session = getattr(callback_context, "session", None)
        _emit_structured_log(
            {
                "severity": "ERROR" if getattr(llm_response, "error_code", None) else "INFO",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "gen_ai.prompt_response",
                "agent_name": getattr(callback_context, "agent_name", "sig_normalizer"),
                "invocation_id": getattr(callback_context, "invocation_id", None),
                "session_id": getattr(session, "id", None),
                "user_id": getattr(callback_context, "user_id", None),
                "model": settings.model,
                "prompt_name": "sig-normalizer",
                "prompt_env": env,
                "prompt_version_pin": pin,
                "user_prompt": user_prompt,
                "model_response": model_response,
                "finish_reason": str(finish_reason) if finish_reason is not None else None,
                "error_code": getattr(llm_response, "error_code", None),
                "error_message": getattr(llm_response, "error_message", None),
                "usage_metadata": usage_dict,
            }
        )
    except Exception:
        pass
    return None


def sig_instruction(ctx: ReadonlyContext) -> str:
    """Resolve the active prompt version for this turn.

    Resolution order:
    1. If session state (or PROMPT_VERSION_PIN env var set by the eval harness)
       carries `prompt_version_pin`, load that exact Vertex AI Prompt Management
       version_id (used by offline evals and canary comparisons).
    2. Otherwise, resolve the release pointer for `prompt_env`
       (defaults to settings.prompt_env, e.g. 'staging' or 'prod').
    """
    state = getattr(ctx, "state", None) or {}
    env = state.get("prompt_env") or os.environ.get("PROMPT_ENV") or settings.prompt_env
    pin = state.get("prompt_version_pin") or os.environ.get("PROMPT_VERSION_PIN")
    return get_prompt_text(
        prompt_name="sig-normalizer",
        env=env,
        version_pin=pin,
    )


# Ensure Vertex AI Evals AgentConfig.from_agent() can inspect ADK agents that use
# a dynamic InstructionProvider callable (google-cloud-aiplatform 2.6.0 passes
# agent.instruction directly to a str field on AgentConfig).
try:
    from vertexai._genai.types import evals as _vertex_evals_types

    _orig_from_agent = _vertex_evals_types.AgentConfig.from_agent.__func__

    @classmethod  # type: ignore[misc]
    def _compatible_from_agent(cls, agent):  # type: ignore[no-untyped-def]
        inst = getattr(agent, "instruction", None)
        if callable(inst):
            env = os.environ.get("PROMPT_ENV") or settings.prompt_env
            pin = os.environ.get("PROMPT_VERSION_PIN")
            resolved_inst = get_prompt_text(
                prompt_name="sig-normalizer",
                env=env,
                version_pin=pin,
            )
            return cls(
                agent_id=getattr(agent, "name", "sig_normalizer"),
                agent_type=agent.__class__.__name__,
                description=getattr(agent, "description", None),
                instruction=resolved_inst,
                tools=cls._get_tool_declarations_from_agent(agent),
                sub_agents=[
                    str(getattr(sub_agent, "name"))
                    for sub_agent in getattr(agent, "sub_agents", [])
                    if getattr(sub_agent, "name", None) is not None
                ],
            )
        return _orig_from_agent(cls, agent)

    _vertex_evals_types.AgentConfig.from_agent = _compatible_from_agent
except Exception:
    pass


root_agent = LlmAgent(
    name="sig_normalizer",
    model=settings.model,  # gemini-3.8-flash (served via global endpoint)
    description=(
        "Normalizes prescription Signatura (Sig) strings into structured JSON "
        "fields, patient-friendly instructions, and pharmacist safety escalations."
    ),
    instruction=sig_instruction,  # Callable -> re-evaluated per invocation
    before_model_callback=log_prompt_request,
    after_model_callback=log_prompt_response,
)
