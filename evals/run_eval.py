#!/usr/bin/env python3
"""Run 100% cloud-based GEAP inference + evaluation against the 8-case Sig dataset and enforce quality gates.

Both steps execute remotely in Google Cloud:
  1. Agent inference runs on the deployed Reasoning Engine in Vertex AI Agent Engine
     (`projects/<project>/locations/<region>/reasoningEngines/<id>`).
  2. Metric evaluation runs on the Vertex AI Evaluation Service (`us-central1`) using
     remote `CodeExecutionMetric`s, `LLMMetric`, and `RubricMetric`s, and publishes
     an `EvaluationExperiment` / `EvaluationRun` linked to the Reasoning Engine.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.genai import types as genai_types
import vertexai
from vertexai import agent_engines
from vertexai._genai import _gcs_utils
from vertexai._genai import types

from config import settings
from evals.metrics import ALL_METRICS
from prompt_registry import _load_pointer_doc, get_prompt_text

EVAL_SET_PATH = ROOT / "evals" / "sig_eval_set.jsonl"
THRESHOLDS_PATH = ROOT / "evals" / "thresholds.yaml"
RESULTS_DIR = ROOT / "results"

AGENT_NAME = "sig_normalizer"
AGENT_DESCRIPTION = (
    "Normalizes prescription Signatura (Sig) strings into structured JSON "
    "fields, patient-friendly instructions, and pharmacist safety escalations."
)

CONSOLE_METRICS = [
    types.RubricMetric.FINAL_RESPONSE_QUALITY(version="v1"),
    types.RubricMetric.FINAL_RESPONSE_MATCH(version="v2"),
    types.RubricMetric.INSTRUCTION_FOLLOWING(version="v1"),
    types.RubricMetric.HALLUCINATION(version="v1"),
]


def _load_eval_rows() -> list[dict[str, Any]]:
    rows = []
    for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_eval_dataframe(
    version_pin: str | None,
    env: str,
) -> pd.DataFrame:
    rows = _load_eval_rows()
    records = []
    for r in rows:
        state = {"prompt_env": env}
        if version_pin is not None:
            state["prompt_version_pin"] = str(version_pin)

        session_inputs = types.evals.SessionInput(
            user_id="eval_runner",
            state=state,
            app_name=AGENT_NAME,
        )
        records.append(
            {
                "case_id": r["case_id"],
                "prompt": r["prompt"],
                "reference": json.dumps(r["reference"]),
                "session_inputs": session_inputs,
            }
        )
    return pd.DataFrame(records)


def _resolve_deployed_agent_engine() -> tuple[str, str]:
    """Return (agent_engine_id, canonical_resource_name) for the deployed sig-normalizer in Google Cloud."""
    vertexai.init(project=settings.project_id, location=settings.resource_location)
    explicit_id = os.getenv("AGENT_ENGINE_ID", "").strip()
    try:
        if explicit_id:
            eng = agent_engines.get(
                f"projects/{settings.project_id}/locations/{settings.resource_location}/reasoningEngines/{explicit_id}"
            )
            ae_id = eng.resource_name.split("/")[-1]
            return (
                ae_id,
                f"projects/{settings.project_id}/locations/{settings.resource_location}/reasoningEngines/{ae_id}",
            )
        for eng in agent_engines.list():
            disp = getattr(eng, "display_name", "") or ""
            if "sig-normalizer" in disp or "sig_normalizer" in disp:
                ae_id = eng.resource_name.split("/")[-1]
                return (
                    ae_id,
                    f"projects/{settings.project_id}/locations/{settings.resource_location}/reasoningEngines/{ae_id}",
                )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve deployed ReasoningEngine in {settings.project_id}/{settings.resource_location}: {exc}"
        ) from exc
    raise RuntimeError(
        f"No deployed 'sig-normalizer' ReasoningEngine found in {settings.project_id}/{settings.resource_location}. "
        "Deploy the agent first (`./deploy_agent.sh sig_normalizer`) so cloud-only evaluation can run."
    )


def _sanitize_resource_name(resource_name: str, agent_engine_id: str = "") -> str:
    """Redact live project ID and ReasoningEngine ID before persisting scorecards to git."""
    sanitized = resource_name
    if settings.project_id:
        sanitized = sanitized.replace(settings.project_id, "your-project-id")
    if agent_engine_id:
        sanitized = sanitized.replace(agent_engine_id, "your-reasoning-engine-id")
    return sanitized


def _ensure_user_event_in_agent_data(
    agent_data: dict[str, Any],
    prompt_text: str,
    response_text: str,
    agents_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure agent_data.turns[0].events starts with a user event, has clean protobuf-compatible parts, and includes agents metadata."""
    ad = copy.deepcopy(agent_data) if isinstance(agent_data, dict) else {}
    turns = ad.get("turns")
    if not isinstance(turns, list) or not turns:
        turns = [{"turn_index": 0, "turn_id": "turn_0", "events": []}]
        ad["turns"] = turns
    first_turn = turns[0]
    raw_events = first_turn.get("events")
    if not isinstance(raw_events, list):
        raw_events = []

    events: list[dict[str, Any]] = []
    for e in raw_events:
        if not isinstance(e, dict):
            continue
        content = e.get("content")
        if not isinstance(content, dict) or not content.get("parts"):
            continue
        cleaned_parts = []
        for p in content.get("parts", []):
            if isinstance(p, dict):
                cp = {
                    k: v
                    for k, v in p.items()
                    if k not in ("thought_signature", "thoughtSignature") and v is not None
                }
                if cp:
                    cleaned_parts.append(cp)
        if cleaned_parts:
            events.append(
                {
                    "author": e.get("author") or AGENT_NAME,
                    "content": {
                        "role": content.get("role", "model"),
                        "parts": cleaned_parts,
                    },
                }
            )

    if not events:
        events = [
            {
                "author": AGENT_NAME,
                "content": {
                    "role": "model",
                    "parts": [{"text": response_text}],
                },
            }
        ]
    if events[0].get("author") != "user":
        events.insert(
            0,
            {
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                },
            },
        )
    first_turn["events"] = events
    if agents_map is not None:
        ad["agents"] = agents_map
    return ad


def _publish_console_evaluation_run(
    client: vertexai.Client,
    *,
    agent_engine_id: str,
    canonical_agent_resource: str,
    label: str,
    version_id: str,
    version_tag: str,
    raw_rows: list[dict[str, Any]],
    agent_dataset: Any,
    resolved_instruction: str,
) -> dict[str, Any] | None:
    """Create a populated EvaluationExperiment, EvaluationSet, and EvaluationRun in Vertex AI Console."""
    bucket_raw = settings.release_bucket.strip()
    if not bucket_raw:
        print("Skipping Console EvaluationRun: PROMPT_RELEASE_BUCKET not configured.")
        return None

    bucket_name = bucket_raw.replace("gs://", "").split("/", 1)[0]
    gcs_prefix = f"gs://{bucket_name}/eval_runs/{label}_{version_tag}"

    # Build AgentInfo with the exact resolved prompt instruction for this version
    agent_config = types.evals.AgentConfig(
        agent_id=AGENT_NAME,
        agent_type="LlmAgent",
        description=AGENT_DESCRIPTION,
        instruction=resolved_instruction,
        tools=[],
        sub_agents=[],
    )
    parsed_agent_info = types.evals.AgentInfo(
        name=AGENT_NAME,
        agents={AGENT_NAME: agent_config},
        root_agent_id=AGENT_NAME,
    )
    agents_map_json = {
        AGENT_NAME: agent_config.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
    }

    inf_df = getattr(agent_dataset, "eval_dataset_df", None)
    if inf_df is None or len(inf_df) == 0:
        print("Skipping Console EvaluationRun: inference dataframe is empty.")
        return None

    gcs_utils = _gcs_utils.GcsUtils(api_client=client.evals._api_client)
    eval_item_names: list[str] = []

    print(
        f">>> Uploading {len(raw_rows)} EvaluationItemRequest payloads to {gcs_prefix}..."
    )
    for idx, row_info in enumerate(raw_rows):
        raw_ad = inf_df.iloc[idx]["agent_data"] if "agent_data" in inf_df.columns else {}
        resp_txt = str(inf_df.iloc[idx]["response"]) if "response" in inf_df.columns else ""
        enriched_ad_dict = _ensure_user_event_in_agent_data(
            raw_ad,
            prompt_text=row_info["prompt"],
            response_text=resp_txt,
            agents_map=agents_map_json,
        )
        agent_data_obj = types.evals.AgentData.model_validate(enriched_ad_dict)

        # IMPORTANT:
        # 1. Only set agent_data on CandidateResponse (leave text=None and events=None)
        #    so the CandidateResponse protobuf oneof (text | events | agent_data) preserves agentData.
        # 2. Omit test_case_title inside UserScenario (set display_name on EvaluationItem instead)
        #    because the backend C++ UserScenario protobuf only accepts starting_prompt and conversation_plan.
        eval_item_request = types.EvaluationItemRequest(
            prompt=types.EvaluationPrompt(
                user_scenario=types.evals.UserScenario(
                    starting_prompt=row_info.get("starting_prompt", row_info["prompt"]),
                    conversation_plan=row_info.get("conversation_plan", row_info["prompt"]),
                )
            ),
            golden_response=types.CandidateResponse(
                text=json.dumps(row_info["reference"])
            ),
            candidate_responses=[
                types.CandidateResponse(
                    candidate=AGENT_NAME,
                    agent_data=agent_data_obj,
                )
            ],
        )
        item_gcs_uri = gcs_utils.upload_json_to_prefix(
            data=eval_item_request.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            gcs_dest_prefix=f"{gcs_prefix}/requests",
            filename_prefix=f"request_{row_info['case_id']}_{uuid.uuid4().hex[:6]}",
        )
        eval_item = client.evals.create_evaluation_item(
            evaluation_item_type=types.EvaluationItemType.REQUEST,
            gcs_uri=item_gcs_uri,
            display_name=row_info.get("test_case_title", row_info["case_id"]),
        )
        eval_item_names.append(eval_item.name)

    eval_set = client.evals.create_evaluation_set(
        evaluation_items=eval_item_names,
        display_name=f"sig-normalizer-{label}-{version_tag}-set",
    )
    set_id = eval_set.name.split("/")[-1]

    # The Python SDK's _CreateEvaluationSetParameters_to_vertex omits agentConfigs;
    # patch agentConfigs onto the EvaluationSet so the Console UI Configuration pane displays Agent details.
    client.evals._api_client.request(
        "patch",
        f"evaluationSets/{set_id}?updateMask=agentConfigs",
        {
            "displayName": f"sig-normalizer-{label}-{version_tag}-set",
            "evaluationItems": eval_item_names,
            "agentConfigs": {
                AGENT_NAME: {
                    "agentId": AGENT_NAME,
                    "description": AGENT_DESCRIPTION,
                    "instruction": resolved_instruction,
                }
            },
        },
    )

    exp_display_name = f"sig-normalizer-{label}-{version_tag}".replace("_", "-")
    # Console UI looks up the experiment's test cases via labels["vertex-ai-evaluation-set-name"]
    eval_labels = {
        "vertex-ai-evaluation-set-name": eval_set.name,
        "vertex-ai-evaluation-agent-engine-id": agent_engine_id,
        "vertex-ai-evaluation-agent-engine-location": settings.resource_location,
        "prompt-version": f"v{version_id}",
        "version-tag": version_tag.lower().replace("_", "-"),
    }

    # Remove any prior experiment with the identical display_name so Console stays clean
    try:
        existing_exps = client.evals.list_evaluation_experiments(
            config={
                "filter": f'labels.vertex-ai-evaluation-agent-engine-id="{agent_engine_id}"'
            }
        )
        for existing in existing_exps.evaluation_experiments or []:
            if existing.display_name == exp_display_name and existing.name:
                client.evals.delete_evaluation_experiment(name=existing.name)
    except Exception:
        pass

    experiment = client.evals.create_evaluation_experiment(
        display_name=exp_display_name,
        labels=eval_labels,
    )

    print(
        f">>> Creating Console EvaluationRun '{exp_display_name}' under experiment {experiment.name.split('/')[-1]}..."
    )
    eval_run = client.evals.create_evaluation_run(
        display_name=exp_display_name,
        evaluation_experiment=experiment.name,
        dataset=types.EvaluationRunDataSource(evaluation_set=eval_set.name),
        dest=f"{gcs_prefix}/results",
        metrics=CONSOLE_METRICS,
        agent_info=parsed_agent_info,
        agent=canonical_agent_resource,
        labels=eval_labels,
        config=types.CreateEvaluationRunConfig(allow_cross_region_model=True),
    )

    run_id = eval_run.name.split("/")[-1]
    exp_id = experiment.name.split("/")[-1]
    set_id = eval_set.name.split("/")[-1]

    print(f">>> Polling EvaluationRun {run_id} until completion...")
    terminal_states = {"SUCCEEDED", "FAILED", "CANCELLED"}
    start_poll = time.time()
    current_state = str(getattr(eval_run, "state", "PENDING"))
    while current_state not in terminal_states and (time.time() - start_poll) < 300:
        time.sleep(5)
        eval_run = client.evals.get_evaluation_run(name=eval_run.name)
        current_state = str(getattr(eval_run, "state", "RUNNING"))
        if hasattr(eval_run.state, "value"):
            current_state = eval_run.state.value
        print(f"    [EvaluationRun {run_id}] state={current_state} ({int(time.time() - start_poll)}s)")

    console_summary: dict[str, float] = {}
    if (
        eval_run.evaluation_run_results
        and eval_run.evaluation_run_results.summary_metrics
        and eval_run.evaluation_run_results.summary_metrics.metrics
    ):
        for k, v in eval_run.evaluation_run_results.summary_metrics.metrics.items():
            if k.endswith("/AVERAGE") and isinstance(v, (int, float)):
                console_summary[k] = round(float(v), 4)

    print(
        f">>> Console EvaluationRun {run_id} finished with state={current_state} | "
        f"Experiment={exp_id} | summary={console_summary}"
    )
    return {
        "experiment_id": exp_id,
        "experiment_display_name": exp_display_name,
        "evaluation_run_id": run_id,
        "evaluation_set_id": set_id,
        "state": current_state,
        "summary_metrics": console_summary,
    }


def _register_candidate_file(
    client: vertexai.Client,
    candidate_path: Path,
) -> tuple[str, str]:
    """Push a candidate prompt file to Vertex AI Prompt Management and return (prompt_id, version_id)."""
    text = candidate_path.read_text(encoding="utf-8").strip()
    doc = _load_pointer_doc("sig-normalizer")
    prompt_id = str(doc["prompt_id"])

    # Check if an existing version in sig-normalizer already has the exact same text
    # so we don't create duplicate versions unnecessarily on repeated CI runs.
    from prompt_registry import _extract_instruction_text

    for v_ref in client.prompts.list_versions(prompt_id=prompt_id):
        vid = str(v_ref.version_id)
        full_v = client.prompts.get_version(prompt_id=prompt_id, version_id=vid)
        if _extract_instruction_text(full_v).strip() == text:
            return prompt_id, vid

    prompt_obj = types.Prompt(
        prompt_data=types.PromptData(
            model=settings.model,
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=text)]
            ),
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text="{sig_input}")],
                )
            ],
        )
    )
    created = client.prompts.update(
        prompt_id=prompt_id,
        prompt=prompt_obj,
        config=types.UpdatePromptConfig(
            prompt_display_name="sig-normalizer",
            version_display_name=f"candidate:{candidate_path.stem}",
        ),
    )
    vid = created._dataset_version.name.split("/")[-1]
    return prompt_id, str(vid)


def evaluate_single_target(
    client: vertexai.Client,
    thresholds: dict[str, Any],
    *,
    version_pin: str | None = None,
    env: str = "staging",
    label_override: str | None = None,
    skip_console_run: bool = False,
) -> dict[str, Any]:
    """Run 100% cloud inference and cloud evaluation for a single prompt version or environment."""
    doc = _load_pointer_doc("sig-normalizer")
    versions_map = doc.get("versions", {})

    resolved_pin = version_pin
    alias = None
    if version_pin is not None:
        pin_str = str(version_pin).strip()
        if pin_str in versions_map:
            resolved_pin = str(versions_map[pin_str]["version_id"])
            alias = pin_str
        else:
            resolved_pin = pin_str[1:] if pin_str.startswith("v") and pin_str[1:].isdigit() else pin_str
            alias = f"v{resolved_pin}"

    agent_engine_id, canonical_agent_resource = _resolve_deployed_agent_engine()
    df = _build_eval_dataframe(version_pin=resolved_pin, env=env)

    resolved_instruction = get_prompt_text(
        prompt_name="sig-normalizer",
        env=env,
        version_pin=resolved_pin,
    )

    print(
        f"\n>>> Running cloud GEAP inference on deployed ReasoningEngine {canonical_agent_resource} "
        f"(target={label_override or alias or f'env:{env}'}, model={settings.model}@{settings.llm_location})..."
    )
    agent_dataset = client.evals.run_inference(
        agent=canonical_agent_resource,
        src=df,
    )
    agent_dataset.candidate_name = AGENT_NAME

    print(
        f">>> Running cloud GEAP evaluation on Vertex AI Evaluation Service across {len(ALL_METRICS)} metrics "
        f"(location={settings.resource_location})..."
    )
    eval_result = client.evals.evaluate(
        dataset=agent_dataset,
        metrics=ALL_METRICS,
    )

    summary_scores: dict[str, float] = {}
    for agg in eval_result.summary_metrics or []:
        summary_scores[agg.metric_name] = round(float(agg.mean_score or 0.0), 4)

    gate_report: dict[str, Any] = {}
    all_passed = True
    for metric_name, rule in thresholds.items():
        actual = summary_scores.get(metric_name, 0.0)
        target = float(rule["min_mean"])
        passed = actual >= target
        all_passed = all_passed and passed
        gate_report[metric_name] = {
            "mean": actual,
            "min_mean": target,
            "passed": passed,
        }

    # Build per-case detail records
    raw_rows = _load_eval_rows()
    inf_df = getattr(agent_dataset, "eval_dataset_df", None)
    case_details: list[dict[str, Any]] = []
    for idx, case_res in enumerate(eval_result.eval_case_results or []):
        row_info = raw_rows[idx] if idx < len(raw_rows) else {"case_id": f"case_{idx}"}
        resp_text = ""
        if inf_df is not None and idx < len(inf_df) and "response" in inf_df.columns:
            resp_val = inf_df.iloc[idx]["response"]
            resp_text = str(resp_val)
        cand_metrics = {}
        if case_res.response_candidate_results:
            for m_name, m_res in (case_res.response_candidate_results[0].metric_results or {}).items():
                cand_metrics[m_name] = {
                    "score": m_res.score,
                    "explanation": m_res.explanation,
                }
        case_details.append(
            {
                "case_id": row_info.get("case_id", f"case_{idx}"),
                "prompt": row_info.get("prompt", ""),
                "response": resp_text,
                "metrics": cand_metrics,
            }
        )

    label = label_override or (alias if alias else f"env_{env}")
    version_id_str = str(resolved_pin or doc["environments"][env]["version_id"])
    version_tag_str = (
        versions_map.get(alias, {}).get("version_tag", label)
        if alias
        else doc["environments"][env]["version_tag"]
    )

    console_run_info = None
    if not skip_console_run:
        console_run_info = _publish_console_evaluation_run(
            client,
            agent_engine_id=agent_engine_id,
            canonical_agent_resource=canonical_agent_resource,
            label=label,
            version_id=version_id_str,
            version_tag=version_tag_str,
            raw_rows=raw_rows,
            agent_dataset=agent_dataset,
            resolved_instruction=resolved_instruction,
        )

    scorecard: dict[str, Any] = {
        "target": label,
        "version_id": version_id_str,
        "version_tag": version_tag_str,
        "agent_engine": _sanitize_resource_name(
            canonical_agent_resource, agent_engine_id
        ),
        "execution_mode": "cloud_only",
        "model": settings.model,
        "llm_location": settings.llm_location,
        "resource_location": settings.resource_location,
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_cases": len(raw_rows),
        "all_gates_passed": all_passed,
        "metrics": gate_report,
        "console_evaluation": console_run_info,
        "cases": case_details,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"scorecard_{label}.json"
    out_file.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    if resolved_pin is not None:
        vid_file = RESULTS_DIR / f"scorecard_v{resolved_pin}.json"
        if vid_file != out_file:
            vid_file.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")

    print("\n=== GEAP Cloud Evaluation Scorecard ===")
    print(
        f"Target: {scorecard['target']} ({scorecard['version_tag']})  |  "
        f"AgentEngine: {agent_engine_id}  |  Model: {scorecard['model']}"
    )
    for m_name, m_data in gate_report.items():
        status = "PASS" if m_data["passed"] else "FAIL"
        print(f"  [{status}] {m_name:30s} mean={m_data['mean']:.2f}  (gate >= {m_data['min_mean']:.2f})")
    print(f"Overall Gate: {'PASSED' if all_passed else 'FAILED'}  -> {out_file}")

    return scorecard


def _render_markdown_scorecard(
    scorecards: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> str:
    ae_res = scorecards[0].get("agent_engine", "") if scorecards else ""
    lines = [
        "# GEAP Hill-Climbing Evaluation Scorecard (Cloud-Only)",
        "",
        f"- **Cloud Agent Engine**: `{ae_res}`",
        f"- **Inference Model**: `{settings.model}` (`{settings.llm_location}`)",
        f"- **GEAP Evaluation Service Region**: `{settings.resource_location}`",
        f"- **Dataset**: `evals/sig_eval_set.jsonl` (8 prescription Signatura cases: 4 standard, 4 high-risk ISMP/clinical safety)",
        "",
    ]

    if len(scorecards) == 1:
        sc = scorecards[0]
        lines.extend(
            [
                f"## Target: `{sc['target']}` (`{sc['version_tag']}`, version_id=`{sc['version_id']}`)",
                "",
                "| Metric | Score | Gate Threshold | Status |",
                "|---|---:|---:|---|",
            ]
        )
        for m_name, m_data in sc["metrics"].items():
            status = "PASS" if m_data["passed"] else "FAIL"
            lines.append(
                f"| `{m_name}` | **{m_data['mean']:.2f}** | `>= {m_data['min_mean']:.2f}` | **{status}** |"
            )
        overall = "PASSED" if sc["all_gates_passed"] else "FAILED"
        lines.extend(["", f"**Overall Gate Status**: **{overall}**", ""])
        return "\n".join(lines)

    # Multi-version hill-climb table
    headers = ["Metric"] + [f"`{sc['target']}` ({sc['version_tag']})" for sc in scorecards] + ["Gate Threshold", "Final Gate Status"]
    sep = ["---"] + ["---:" for _ in scorecards] + ["---:", "---"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    final_sc = scorecards[-1]
    for m_name, rule in thresholds.items():
        row_cells = [f"`{m_name}`"]
        for sc in scorecards:
            val = sc["metrics"][m_name]["mean"]
            row_cells.append(f"**{val:.2f}**" if sc is final_sc else f"{val:.2f}")
        row_cells.append(f"`>= {float(rule['min_mean']):.2f}`")
        status = "PASS" if final_sc["metrics"][m_name]["passed"] else "FAIL"
        row_cells.append(f"**{status}**")
        lines.append("| " + " | ".join(row_cells) + " |")

    gate_row = ["**Promotion Gate**"]
    for sc in scorecards:
        gate_row.append("**PASS (Promotable)**" if sc["all_gates_passed"] else "FAIL (Blocked)")
    gate_row.extend(["All 3 metrics pass", "**PASSED**" if final_sc["all_gates_passed"] else "**FAILED**"])
    lines.append("| " + " | ".join(gate_row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="Pin a specific version (e.g. v1, v2, v3, or 1, 2, 3)")
    parser.add_argument("--env", default="staging", help="Environment pointer to evaluate when --version is omitted")
    parser.add_argument("--candidate-file", default=None, help="Local prompt file to register and evaluate")
    parser.add_argument("--baseline-env", default=None, help="Optional baseline environment to compare against")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Evaluate and compare multiple versions side-by-side (e.g. --compare v1 v2 v3)",
    )
    parser.add_argument("--markdown-out", default=None, help="Optional path to write Markdown scorecard")
    parser.add_argument(
        "--skip-console-run",
        action="store_true",
        help="Skip publishing EvaluationExperiment/EvaluationRun to the Google Cloud Console",
    )
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="Exit 0 even if quality gates fail (useful when recording baseline failure scorecards)",
    )
    args = parser.parse_args()

    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))["metrics"]
    client = vertexai.Client(
        project=settings.project_id,
        location=settings.resource_location,
    )

    if args.compare:
        scorecards = []
        for target_ver in args.compare:
            sc = evaluate_single_target(
                client,
                thresholds,
                version_pin=target_ver,
                env=args.env,
                skip_console_run=args.skip_console_run,
            )
            scorecards.append(sc)

        summary_doc = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "execution_mode": "cloud_only",
            "agent_engine": scorecards[0].get("agent_engine") if scorecards else None,
            "model": settings.model,
            "llm_location": settings.llm_location,
            "resource_location": settings.resource_location,
            "thresholds": thresholds,
            "runs": [
                {
                    "target": sc["target"],
                    "version_id": sc["version_id"],
                    "version_tag": sc["version_tag"],
                    "all_gates_passed": sc["all_gates_passed"],
                    "metrics": sc["metrics"],
                    "console_evaluation": sc.get("console_evaluation"),
                }
                for sc in scorecards
            ],
        }
        summary_path = RESULTS_DIR / "hill_climb_summary.json"
        summary_path.write_text(json.dumps(summary_doc, indent=2) + "\n", encoding="utf-8")

        md_text = _render_markdown_scorecard(scorecards, thresholds)
        md_path = Path(args.markdown_out) if args.markdown_out else (RESULTS_DIR / "hill_climb_scorecard.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8")
        print(f"\nWrote Hill-Climb Summary JSON -> {summary_path}")
        print(f"Wrote Hill-Climb Scorecard MD -> {md_path}\n")
        print(md_text)

        if not scorecards[-1]["all_gates_passed"] and not args.allow_gate_failure:
            sys.exit(1)
        return

    scorecards_to_render = []
    if args.baseline_env:
        base_sc = evaluate_single_target(
            client,
            thresholds,
            version_pin=None,
            env=args.baseline_env,
            label_override=f"baseline_{args.baseline_env}",
            skip_console_run=args.skip_console_run,
        )
        scorecards_to_render.append(base_sc)

    version_pin = args.version
    label_override = None
    if args.candidate_file:
        cand_path = Path(args.candidate_file)
        _, version_pin = _register_candidate_file(client, cand_path)
        label_override = f"v{version_pin}"

    sc = evaluate_single_target(
        client,
        thresholds,
        version_pin=version_pin,
        env=args.env,
        label_override=label_override,
        skip_console_run=args.skip_console_run,
    )
    scorecards_to_render.append(sc)

    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_render_markdown_scorecard(scorecards_to_render, thresholds), encoding="utf-8")
        print(f"Wrote Markdown scorecard -> {md_path}")

    if not sc["all_gates_passed"] and not args.allow_gate_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
