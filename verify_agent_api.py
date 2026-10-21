#!/usr/bin/env python3
"""Verify the deployed Sig Normalizer agent on Vertex AI Agent Runtime via REST API."""

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
LOCATION = os.getenv("AGENT_LOCATION") or os.getenv("RESOURCE_LOCATION") or "us-central1"
if LOCATION == "global":
    LOCATION = "us-central1"

if not PROJECT or PROJECT == "your-project-id":
    print("Error: GOOGLE_CLOUD_PROJECT not set in .env", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", default=None, help="Reasoning Engine ID (auto-detected if omitted)")
    parser.add_argument(
        "--prompt",
        default="metformin 500mg tab: 1 tab po bid c meals",
        help="Prescription Sig string to send to the deployed agent",
    )
    parser.add_argument(
        "--prompt-env",
        default=None,
        help="Optional session state override for prompt_env ('prod' or 'staging')",
    )
    parser.add_argument(
        "--version-pin",
        default=None,
        help="Optional session state override for prompt_version_pin (e.g. '1', '2', '3')",
    )
    args = parser.parse_args()

    agent_id = args.agent_id
    if not agent_id:
        try:
            agent_id = subprocess.check_output(
                [sys.executable, "get_agent_id.py", "sig_normalizer"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            agent_id = subprocess.check_output(
                [sys.executable, "get_agent_id.py"],
                text=True,
            ).strip()

    print(f"Testing Reasoning Engine ID: {agent_id} (Region: {LOCATION})")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())

    base_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/reasoningEngines/{agent_id}"
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
        "x-goog-user-project": PROJECT,
    }

    state = {}
    if args.prompt_env:
        state["prompt_env"] = args.prompt_env
    if args.version_pin:
        state["prompt_version_pin"] = str(args.version_pin)

    print("1. Creating session...")
    session_input = {"user_id": "verify_script_user"}
    if state:
        session_input["state"] = state

    session_resp = requests.post(
        f"{base_url}:query",
        headers=headers,
        json={
            "class_method": "async_create_session",
            "input": session_input,
        },
        timeout=60,
    )
    if not session_resp.ok:
        print(f"Failed to create session ({session_resp.status_code}): {session_resp.text}", file=sys.stderr)
        sys.exit(1)

    session_data = session_resp.json()
    session_id = session_data.get("output", {}).get("id")
    print(f"   Session created: {session_id} (state={state or 'default'})")

    print(f"2. Sending Sig query: {args.prompt!r}...")
    stream_resp = requests.post(
        f"{base_url}:streamQuery",
        headers=headers,
        json={
            "class_method": "async_stream_query",
            "input": {
                "user_id": "verify_script_user",
                "session_id": session_id,
                "message": args.prompt,
            },
        },
        stream=True,
        timeout=120,
    )

    if not stream_resp.ok:
        print(f"Stream query failed ({stream_resp.status_code}): {stream_resp.text}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Agent Response ---")
    for line in stream_resp.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            try:
                event = json.loads(decoded)
                parts = event.get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        print(part["text"], end="", flush=True)
            except json.JSONDecodeError:
                print(decoded)
    print("\n----------------------")


if __name__ == "__main__":
    main()
