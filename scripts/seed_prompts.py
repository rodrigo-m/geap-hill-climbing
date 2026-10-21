#!/usr/bin/env python3
"""Register v1, v2, and v3 of sig-normalizer in Vertex AI Prompt Management."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.cloud import storage
from google.genai import types as genai_types
import vertexai
from vertexai._genai import types

from config import settings

PROMPTS_DIR = ROOT / "prompts"
RELEASE_FILE = ROOT / "release" / "sig-normalizer.json"
PKG_RELEASE_FILE = ROOT / "sig_normalizer" / "release" / "sig-normalizer.json"

VERSIONS = [
    ("v1", "v1_naive", "v1: naive baseline", PROMPTS_DIR / "v1_naive.txt"),
    ("v2", "v2_schema", "v2: strict JSON schema + abbrev map", PROMPTS_DIR / "v2_schema.txt"),
    ("v3", "v3_policy_fewshot", "v3: ISMP safety policy + few-shot", PROMPTS_DIR / "v3_policy_fewshot.txt"),
]


def _build_prompt(instruction_text: str) -> types.Prompt:
    return types.Prompt(
        prompt_data=types.PromptData(
            model=settings.model,
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=instruction_text)]
            ),
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="{sig_input}")],
                )
            ],
        ),
    )


def _sync_release_to_gcs(pointer_doc: dict) -> None:
    if not settings.release_bucket or "your-project-id" in settings.release_bucket:
        return
    bucket_name = settings.release_bucket.replace("gs://", "").split("/")[0]
    gcs = storage.Client(project=settings.project_id or None)
    bucket = gcs.bucket(bucket_name)
    if not bucket.exists():
        print(f"Creating release bucket gs://{bucket_name} in {settings.resource_location}...")
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.iam_configuration.public_access_prevention = "enforced"
        bucket = gcs.create_bucket(bucket, location=settings.resource_location)
    blob = bucket.blob("release/sig-normalizer.json")
    blob.upload_from_string(
        json.dumps(pointer_doc, indent=2) + "\n",
        content_type="application/json",
    )
    print(f"Uploaded release pointer to gs://{bucket_name}/release/sig-normalizer.json")


def main() -> None:
    if not settings.project_id or settings.project_id == "your-project-id":
        raise SystemExit("ERROR: Set GOOGLE_CLOUD_PROJECT in .env before running seed_prompts.py")

    client = vertexai.Client(
        project=settings.project_id,
        location=settings.resource_location,
    )

    prompt_id: str | None = None
    version_map: dict[str, dict[str, str]] = {}

    for idx, (alias, tag, display_note, path) in enumerate(VERSIONS):
        text = path.read_text(encoding="utf-8").strip()
        prompt_obj = _build_prompt(text)

        if idx == 0:
            created = client.prompts.create_version(
                prompt=prompt_obj,
                config=types.CreatePromptVersionConfig(
                    prompt_display_name="sig-normalizer",
                    version_display_name=display_note,
                ),
            )
            prompt_id = created._dataset.name.split("/")[-1]
        else:
            assert prompt_id is not None
            created = client.prompts.update(
                prompt_id=prompt_id,
                prompt=prompt_obj,
                config=types.UpdatePromptConfig(
                    prompt_display_name="sig-normalizer",
                    version_display_name=display_note,
                ),
            )

        version_id = created._dataset_version.name.split("/")[-1]
        version_map[alias] = {
            "version_id": str(version_id),
            "version_tag": tag,
            "display_name": display_note,
        }
        print(
            f"Registered {tag:20s} -> prompt_id={prompt_id}  version_id={version_id}"
        )

    # Also register a scratch prompt resource for CI / local candidate-file testing
    scratch_created = client.prompts.create_version(
        prompt=_build_prompt(VERSIONS[-1][3].read_text(encoding="utf-8").strip()),
        config=types.CreatePromptVersionConfig(
            prompt_display_name="sig-normalizer-scratch",
            version_display_name="scratch-init",
        ),
    )
    scratch_prompt_id = scratch_created._dataset.name.split("/")[-1]
    print(f"Registered scratch prompt   -> scratch_prompt_id={scratch_prompt_id}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pointer_doc = {
        "prompt_name": "sig-normalizer",
        "prompt_id": prompt_id,
        "scratch_prompt_id": scratch_prompt_id,
        "resource_location": settings.resource_location,
        "versions": version_map,
        "environments": {
            "prod": {
                "version_id": version_map["v1"]["version_id"],
                "version_tag": version_map["v1"]["version_tag"],
                "updated_at": now,
                "updated_by": "seed_prompts.py",
            },
            "staging": {
                "version_id": version_map["v3"]["version_id"],
                "version_tag": version_map["v3"]["version_tag"],
                "updated_at": now,
                "updated_by": "seed_prompts.py",
            },
        },
    }

    RELEASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RELEASE_FILE.write_text(json.dumps(pointer_doc, indent=2) + "\n", encoding="utf-8")
    PKG_RELEASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PKG_RELEASE_FILE.write_text(json.dumps(pointer_doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote release pointer -> {RELEASE_FILE}")

    _sync_release_to_gcs(pointer_doc)


if __name__ == "__main__":
    main()
