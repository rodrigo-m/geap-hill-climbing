#!/usr/bin/env python3
"""Fetch recent Cloud Logging entries (including OTel & prompt-response logs) for a deployed Reasoning Engine."""

import argparse
import json
import os
import subprocess
import sys

from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import Request
import requests

load_dotenv()

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT or PROJECT == "your-project-id":
    print("Error: GOOGLE_CLOUD_PROJECT not set in .env", file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(
    description="Fetch Cloud Logging entries and prompt-response telemetry for a deployed Reasoning Engine."
)
parser.add_argument(
    "agent_target",
    nargs="?",
    default=None,
    help="Reasoning Engine ID or agent folder name (e.g. sig_normalizer).",
)
parser.add_argument(
    "pos_limit",
    nargs="?",
    type=int,
    default=None,
    help="Optional positional limit on number of log entries to return.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=30,
    help="Maximum number of log entries to return (default: 30).",
)
parser.add_argument(
    "--prompt-response-only",
    action="store_true",
    help="Filter only OpenTelemetry GenAI and structured prompt-response log events.",
)
args = parser.parse_args()

limit = args.pos_limit if args.pos_limit is not None else args.limit

if args.agent_target and args.agent_target.isdigit():
    AGENT_ID = args.agent_target
else:
    cmd = [sys.executable, "get_agent_id.py"]
    if args.agent_target:
        cmd.append(args.agent_target)
    AGENT_ID = subprocess.check_output(cmd, text=True).strip()

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(Request())

log_filter = (
    f'resource.type="aiplatform.googleapis.com/ReasoningEngine" '
    f'AND resource.labels.reasoning_engine_id="{AGENT_ID}"'
)
if args.prompt_response_only:
    log_filter += (
        ' AND ('
        'logName:("gen_ai." OR "gcp.vertex.agent") '
        'OR labels."opentelemetry.io/event.name":("gen_ai." OR "gcp.vertex.agent") '
        'OR jsonPayload.event_type:("gen_ai.prompt_request" OR "gen_ai.prompt_response") '
        'OR jsonPayload.content.role:("user" OR "model") '
        'OR textPayload:"gen_ai.prompt_"'
        ')'
    )

url = "https://logging.googleapis.com/v2/entries:list"
headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json",
    "x-goog-user-project": PROJECT,
}
payload = {
    "resourceNames": [f"projects/{PROJECT}"],
    "filter": log_filter,
    "orderBy": "timestamp desc",
    "pageSize": limit,
}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
if not resp.ok:
    print(f"Logging API Error ({resp.status_code}): {resp.text}", file=sys.stderr)
    sys.exit(1)

entries = resp.json().get("entries", [])
if not entries:
    print(f"No matching log entries found for Reasoning Engine {AGENT_ID}.")
    sys.exit(0)


def _clean_payload(obj):
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k in ("thought_signature", "thoughtSignature") and isinstance(v, str) and len(v) > 32:
                cleaned[k] = f"<{len(v)} chars>"
            elif v is not None:
                cleaned[k] = _clean_payload(v)
        return cleaned
    if isinstance(obj, list):
        return [_clean_payload(item) for item in obj]
    return obj


for entry in reversed(entries):
    ts = entry.get("timestamp", "")
    severity = entry.get("severity", "INFO")
    labels = entry.get("labels") or {}
    log_name = entry.get("logName", "").split("/logs/")[-1]
    otel_event = labels.get("opentelemetry.io/event.name", "") or (
        log_name if log_name.startswith("gen_ai.") else ""
    )
    trace_path = entry.get("trace", "")
    trace_short = trace_path.split("/")[-1][:12] if trace_path else ""
    prefix = f"[{ts}] [{severity}]"
    if trace_short:
        prefix += f" [trace:{trace_short}]"
    if otel_event:
        prefix += f" [otel:{otel_event}]"

    if "jsonPayload" in entry:
        msg = json.dumps(_clean_payload(entry["jsonPayload"]), ensure_ascii=False)
    else:
        msg = entry.get("textPayload", "").strip()
    print(f"{prefix} {msg}")
