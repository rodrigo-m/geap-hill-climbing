"""Runtime prompt resolver with TTL cache and release-pointer lookup."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

from google.cloud import storage
import vertexai

try:
    from .config import settings
except ImportError:
    from config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReleasePointer:
    prompt_name: str
    prompt_id: str
    env: str
    version_id: str
    version_tag: str


class _TTLCache:
    """Thread-safe in-process TTL cache so the agent does not hit the API every turn."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_pointer_cache = _TTLCache(settings.prompt_cache_ttl_seconds)
_prompt_text_cache = _TTLCache(settings.prompt_cache_ttl_seconds)


def _find_local_pointer_path(prompt_name: str) -> Path:
    """Locate release/<prompt_name>.json relative to module or parent repo root."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "release" / f"{prompt_name}.json",
        here.parent / "release" / f"{prompt_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_pointer_doc(prompt_name: str) -> dict[str, Any]:
    """Load release/<prompt_name>.json from GCS (if configured) with local fallback."""
    if settings.release_bucket and "your-project-id" not in settings.release_bucket:
        try:
            bucket_name = settings.release_bucket.replace("gs://", "").split("/")[0]
            blob_path = f"release/{prompt_name}.json"
            gcs = storage.Client(project=settings.project_id or None)
            blob = gcs.bucket(bucket_name).blob(blob_path)
            if blob.exists():
                return json.loads(blob.download_as_text())
        except Exception as exc:
            logger.warning(
                "Falling back to local release pointer after GCS read warning: %s",
                exc,
            )
    local_path = _find_local_pointer_path(prompt_name)
    return json.loads(local_path.read_text(encoding="utf-8"))


def resolve_pointer(
    prompt_name: str,
    env: str,
    version_pin: str | None = None,
) -> ReleasePointer:
    """Resolve (prompt_name, env[, version_pin]) -> immutable Vertex AI version_id."""
    cache_key = f"{prompt_name}:{env}:{version_pin or 'auto'}"
    cached = _pointer_cache.get(cache_key)
    if cached is not None:
        return cached

    doc = _load_pointer_doc(prompt_name)
    prompt_id = str(doc["prompt_id"])
    versions_map = doc.get("versions", {})

    if version_pin is not None:
        pin_str = str(version_pin).strip()
        if pin_str in versions_map:
            ver_info = versions_map[pin_str]
            pointer = ReleasePointer(
                prompt_name=prompt_name,
                prompt_id=str(ver_info.get("prompt_id", prompt_id)),
                env=f"pinned:{pin_str}",
                version_id=str(ver_info["version_id"]),
                version_tag=str(ver_info.get("version_tag", pin_str)),
            )
        else:
            # Normalize e.g. "v1" -> "1" if versions_map is not populated
            normalized_id = pin_str[1:] if pin_str.startswith("v") and pin_str[1:].isdigit() else pin_str
            pointer = ReleasePointer(
                prompt_name=prompt_name,
                prompt_id=prompt_id,
                env=f"pinned:{pin_str}",
                version_id=normalized_id,
                version_tag=f"pinned-{pin_str}",
            )
    else:
        env_entry = doc["environments"][env]
        pointer = ReleasePointer(
            prompt_name=prompt_name,
            prompt_id=str(env_entry.get("prompt_id", prompt_id)),
            env=env,
            version_id=str(env_entry["version_id"]),
            version_tag=str(env_entry["version_tag"]),
        )

    _pointer_cache.set(cache_key, pointer)
    return pointer


def _extract_instruction_text(prompt_obj: Any) -> str:
    """Extract system instruction / prompt text from a versioned Vertex AI Prompt object.

    Note: In google-cloud-aiplatform 2.6.0, client.prompts.get_version() attaches the
    version-specific DatasetVersion resource to prompt_obj._dataset_version while
    populating prompt_obj.prompt_data from the parent Dataset metadata. We inspect
    _dataset_version first so version pinning always returns the exact version text.
    """
    dataset_version = getattr(prompt_obj, "_dataset_version", None)
    if dataset_version is not None:
        try:
            pm = dataset_version.metadata.prompt_api_schema.multimodal_prompt.prompt_message
            if pm is not None:
                if pm.system_instruction and pm.system_instruction.parts:
                    text = "".join(
                        p.text for p in pm.system_instruction.parts if getattr(p, "text", None)
                    )
                    if text.strip():
                        return text
                if pm.contents:
                    for content in pm.contents:
                        if content.parts:
                            text = "".join(
                                p.text for p in content.parts if getattr(p, "text", None)
                            )
                            if text.strip():
                                return text
        except Exception as exc:
            logger.debug("Could not extract from _dataset_version metadata: %s", exc)

    data = getattr(prompt_obj, "prompt_data", None)
    if data is not None:
        if getattr(data, "system_instruction", None) and data.system_instruction.parts:
            text = "".join(
                p.text for p in data.system_instruction.parts if getattr(p, "text", None)
            )
            if text.strip():
                return text
        if getattr(data, "contents", None):
            for content in data.contents:
                if content.parts:
                    text = "".join(
                        p.text for p in content.parts if getattr(p, "text", None)
                    )
                    if text.strip():
                        return text

    if hasattr(prompt_obj, "assemble_contents"):
        parts = prompt_obj.assemble_contents()
        return "\n".join(
            p.text for c in parts for p in getattr(c, "parts", []) if getattr(p, "text", None)
        )

    raise ValueError(f"Unable to extract instruction text from Prompt object: {prompt_obj}")


def get_prompt_text(
    prompt_name: str = "sig-normalizer",
    env: str | None = None,
    version_pin: str | None = None,
) -> str:
    """Fetch the active system instruction string from Vertex AI Prompt Management."""
    target_env = env or settings.prompt_env
    pointer = resolve_pointer(prompt_name, target_env, version_pin=version_pin)

    text_key = f"{pointer.prompt_id}:{pointer.version_id}"
    cached_text = _prompt_text_cache.get(text_key)
    if cached_text is not None:
        return cached_text

    client = vertexai.Client(
        project=settings.project_id,
        location=settings.resource_location,
    )
    prompt_version = client.prompts.get_version(
        prompt_id=pointer.prompt_id,
        version_id=pointer.version_id,
    )

    instruction_text = _extract_instruction_text(prompt_version)

    logger.info(
        "Loaded prompt name=%s env=%s prompt_id=%s version_id=%s tag=%s",
        pointer.prompt_name,
        pointer.env,
        pointer.prompt_id,
        pointer.version_id,
        pointer.version_tag,
    )
    _prompt_text_cache.set(text_key, instruction_text)
    return instruction_text
