#!/usr/bin/env python3
"""Inspect registered versions of sig-normalizer in Vertex AI Prompt Management."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vertexai

from config import settings
from prompt_registry import _extract_instruction_text, _load_pointer_doc


def main() -> None:
    doc = _load_pointer_doc("sig-normalizer")
    prompt_id = str(doc["prompt_id"])
    envs = doc.get("environments", {})

    client = vertexai.Client(
        project=settings.project_id,
        location=settings.resource_location,
    )

    print(f"Prompt Resource: sig-normalizer (prompt_id={prompt_id}, location={settings.resource_location})")
    print("-" * 88)
    print(f"{'Version ID':<12} {'Display Name':<38} {'Active Environments':<22} {'Chars':<8}")
    print("-" * 88)

    versions = list(client.prompts.list_versions(prompt_id=prompt_id))
    for v_ref in versions:
        vid = str(v_ref.version_id)
        full_v = client.prompts.get_version(prompt_id=prompt_id, version_id=vid)
        dv = getattr(full_v, "_dataset_version", None)
        display_name = getattr(dv, "display_name", None) or vid
        text = _extract_instruction_text(full_v)
        active_in = [
            env_name
            for env_name, env_cfg in envs.items()
            if str(env_cfg.get("version_id")) == vid
        ]
        active_str = ", ".join(active_in) if active_in else "-"
        print(f"{vid:<12} {display_name:<38} {active_str:<22} {len(text):<8}")

    print("-" * 88)
    print("Release Pointer Summary:")
    print(json.dumps(envs, indent=2))


if __name__ == "__main__":
    main()
