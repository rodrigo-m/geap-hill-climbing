#!/usr/bin/env python3
"""Promote a prompt version into staging or prod by updating the release pointer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.cloud import storage
import vertexai

from config import settings
from prompt_registry import _pointer_cache, _prompt_text_cache

RELEASE_FILE = ROOT / "release" / "sig-normalizer.json"
PKG_RELEASE_FILE = ROOT / "sig_normalizer" / "release" / "sig-normalizer.json"
RESULTS_DIR = ROOT / "results"


def _sync_release_to_gcs(doc: dict) -> None:
    if not settings.release_bucket or "your-project-id" in settings.release_bucket:
        return
    bucket_name = settings.release_bucket.replace("gs://", "").split("/")[0]
    gcs = storage.Client(project=settings.project_id or None)
    blob = gcs.bucket(bucket_name).blob("release/sig-normalizer.json")
    blob.upload_from_string(
        json.dumps(doc, indent=2) + "\n",
        content_type="application/json",
    )
    print(f"Synced release pointer to gs://{bucket_name}/release/sig-normalizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["staging", "prod"], required=True)
    parser.add_argument(
        "--to-version",
        required=True,
        help="Target Vertex AI version_id (e.g. '3') or alias (e.g. 'v3')",
    )
    parser.add_argument(
        "--version-tag",
        default=None,
        help="Optional human-readable tag (auto-resolved from release file if omitted)",
    )
    parser.add_argument(
        "--skip-eval-check",
        action="store_true",
        help="Bypass local scorecard gate verification (for emergency rollback)",
    )
    args = parser.parse_args()

    doc = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    versions_map = doc.get("versions", {})

    raw_target = str(args.to_version).strip()
    if raw_target in versions_map:
        version_id = str(versions_map[raw_target]["version_id"])
        default_tag = str(versions_map[raw_target]["version_tag"])
        alias = raw_target
    else:
        version_id = raw_target[1:] if raw_target.startswith("v") and raw_target[1:].isdigit() else raw_target
        alias = f"v{version_id}"
        default_tag = versions_map.get(alias, {}).get("version_tag", f"v{version_id}")

    version_tag = args.version_tag or default_tag

    # 1. Verify the version exists in Vertex AI Prompt Management
    client = vertexai.Client(
        project=settings.project_id,
        location=settings.resource_location,
    )
    _ = client.prompts.get_version(
        prompt_id=str(doc["prompt_id"]),
        version_id=version_id,
    )

    # 2. Require a passing evaluation scorecard for the target version
    if not args.skip_eval_check:
        candidates = [
            RESULTS_DIR / f"scorecard_{alias}.json",
            RESULTS_DIR / f"scorecard_v{version_id}.json",
        ]
        scorecard_path = next((p for p in candidates if p.exists()), None)
        if scorecard_path is None:
            raise SystemExit(
                f"REFUSED: No evaluation scorecard found at {candidates[0]}. "
                f"Run `uv run python evals/run_eval.py --version {alias}` first."
            )
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        if not scorecard.get("all_gates_passed"):
            raise SystemExit(
                f"REFUSED: Version {version_id} ({version_tag}) failed one or more "
                f"quality gates in {scorecard_path.name}. Promotion blocked."
            )

    prev = doc["environments"].get(args.env, {})
    doc["environments"][args.env] = {
        "version_id": version_id,
        "version_tag": version_tag,
        "previous_version_id": prev.get("version_id"),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_by": "promote.py",
    }

    RELEASE_FILE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    if PKG_RELEASE_FILE.parent.exists():
        PKG_RELEASE_FILE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    _sync_release_to_gcs(doc)
    _pointer_cache.clear()
    _prompt_text_cache.clear()

    print(
        f"Promoted env={args.env} from version={prev.get('version_id')} "
        f"-> version={version_id} ({version_tag})"
    )


if __name__ == "__main__":
    main()
