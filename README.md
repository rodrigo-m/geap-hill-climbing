# Prescription Sig Normalizer — Prompt Versioning & Evaluation-Gated Hill-Climbing on Vertex AI

A reference implementation of **systematic prompt engineering ("hill-climbing")** and **decoupled prompt release management** for a clinical **Google Agent Development Kit (ADK)** agent running on **Vertex AI Agent Runtime (Reasoning Engine)**, powered by **Gemini 3.8 Flash** (`global`) and **Vertex AI GenAI Evaluation & Prompt Management** (`us-central1`).

> **Quickstart & Setup**: See **[`INSTALLATION.md`](INSTALLATION.md)** for step-by-step environment setup, prompt seeding, evaluation execution, and Vertex AI Agent Runtime deployment instructions.

---

## 1. Problem Statement: Why Hill-Climbing Needs Evaluation Gates

Retail and hospital pharmacies receive millions of prescription **Signatura ("Sig")** strings written in dense medical shorthand:

- `metformin 500mg tab: 1 tab po bid c meals`
- `ofloxacin 0.3% otic soln: 5 gtt au bid x 7d`
- `insulin glargine: 10 u subq qhs` *(contains ISMP "Do Not Use" abbreviation `u` — 10x overdose hazard)*
- `methotrexate 2.5mg tab: 1 tab po daily` *(fatal daily-vs-weekly schedule error for non-oncology oral methotrexate)*

A **Sig Normalizer Agent** must simultaneously satisfy three competing requirements:
1. **Strict Schema Compliance (`structured_field_accuracy`)**: Emit machine-parseable JSON (`drug`, `dose`, `route`, `frequency`, `duration_days`, `prn_reason`, `needs_pharmacist_review`) for downstream pharmacy dispensing software.
2. **Clinical & ISMP Safety Escalation (`safety_escalation`)**: Detect Institute for Safe Medication Practices (ISMP) error-prone abbreviations (`U`, `IU`, `QD`, `QOD`, trailing zero `5.0 mg`, naked decimal `.5 mg`), supratherapeutic daily cumulative doses (`acetaminophen > 4,000 mg/day`), and high-alert schedule errors (`daily` oral methotrexate), setting `needs_pharmacist_review: true` and withholding the unverified dose (`dose: null`).
3. **Patient-Friendly Literacy (`patient_instruction_quality`)**: Produce 6th-grade reading-level label instructions in plain English with zero raw Latin abbreviations—or an explicit `"HOLD LABEL"` directive when pharmacist verification is required.

Improving one axis (e.g., forcing strict JSON output) often leaves blind spots on another (e.g., silently normalizing a dangerous abbreviation instead of escalating). **Evaluation-gated hill-climbing** turns prompt iteration into a measurable, regression-proof engineering discipline.

---

## 2. Architecture: Decoupling Prompt Releases from Agent Redeployment

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Git Repository (Source of Truth & Governance)                                │
│  prompts/v1_naive.txt  prompts/v2_schema.txt  prompts/v3_policy_fewshot.txt  │
│  evals/sig_eval_set.jsonl    evals/thresholds.yaml    release/sig-normalizer │
└───────────────┬──────────────────────────────────────────────┬───────────────┘
                │ scripts/seed_prompts.py                      │ evals/run_eval.py
                ▼                                              ▼
┌──────────────────────────────────────────┐  ┌────────────────────────────────┐
│ Vertex AI Prompt Management (us-central1)│  │ Vertex AI GenAI Evaluation     │
│  client.prompts.create_version / update  │  │  client.evals.run_inference    │
│  Dataset ID + Immutable DatasetVersions  │  │  client.evals.evaluate         │
│   • Version 1: v1_naive                  │  │   • structured_field_accuracy  │
│   • Version 2: v2_schema                 │  │   • safety_escalation          │
│   • Version 3: v3_policy_fewshot         │  │   • patient_instruction_quality│
│                                          │  │  Console EvaluationExperiment  │
│                                          │  │   & EvaluationRun (Rubrics)    │
└───────────────┬──────────────────────────┘  └────────────────┬───────────────┘
                │ client.prompts.get_version                   │ Gates scripts/promote.py
                ▼                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Vertex AI Agent Runtime (ReasoningEngine in us-central1)                     │
│  • root_agent = LlmAgent(model="gemini-3.8-flash", instruction=sig_instruction)
│  • sig_instruction(ctx: ReadonlyContext) -> str resolves active version_id   │
│    via release/sig-normalizer.json (or GCS replica) with a 60s TTL cache     │
│  • LLM inference routed to gemini-3.8-flash in the 'global' multiregion      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Patterns

1. **Dynamic `InstructionProvider` Callable ([`agent.py`](agent.py), [`sig_normalizer/agent.py`](sig_normalizer/agent.py))**:
   Instead of baking a static prompt string into `LlmAgent(instruction="...")` at container build time, `root_agent` passes a callable `sig_instruction(ctx: ReadonlyContext) -> str`. On each turn, the agent resolves the active environment (`prod` or `staging`) or an evaluation pin (`prompt_version_pin`), fetches the immutable prompt version from **Vertex AI Prompt Management** via [`prompt_registry.py`](prompt_registry.py), and caches it in memory with a 60-second TTL.
2. **Two-Tier Version Identity ([`release/sig-normalizer.json`](release/sig-normalizer.json))**:
   - **Immutable server-side `version_id`** (`"1"`, `"2"`, `"3"`) stored as `DatasetVersion` resources in Vertex AI Prompt Management.
   - **Environment release pointers** (`prod`, `staging`) stored in [`release/sig-normalizer.json`](release/sig-normalizer.json) and synced to Cloud Storage (`PROMPT_RELEASE_BUCKET`). Promoting or rolling back a prompt between `staging` and `prod` is a pointer update—**zero container rebuild or Reasoning Engine redeploy required**.
3. **Global Inference + Regional Control Plane ([`config.py`](config.py))**:
   - **Inference**: `gemini-3.8-flash` is served from the `global` multiregion endpoint (`LLM_LOCATION=global`, `GOOGLE_CLOUD_LOCATION=global`).
   - **Prompt Registry, Agent Runtime & Evaluation**: Vertex AI Prompt Management, Agent Runtime (`ReasoningEngine`), and GenAI Evaluation APIs run in `us-central1` (`RESOURCE_LOCATION=us-central1`).
4. **100% Cloud-Executed Evaluation & Console Experiment Tracking ([`evals/run_eval.py`](evals/run_eval.py), [`evals/metrics.py`](evals/metrics.py))**:
   - **Remote Agent Inference (`client.evals.run_inference`)**: Executes all 8 evaluation cases directly against the deployed `ReasoningEngine` in Vertex AI Agent Runtime, passing `session_inputs` (`prompt_env` and `prompt_version_pin`) per case to evaluate any server-side prompt version on the live cloud agent.
   - **Remote Metric Evaluation (`client.evals.evaluate`)**: Scores agent responses on the Vertex AI Evaluation Service using two cloud Python sandbox metrics (`types.CodeExecutionMetric`) and one clinical pharmacist LLM-as-a-Judge autorater (`types.LLMMetric`).
   - **Google Cloud Console Publishing (`create_evaluation_experiment` / `create_evaluation_run`)**: Automatically uploads `EvaluationItemRequest` payloads to GCS, creates an `EvaluationSet` and `EvaluationExperiment` linked to the Reasoning Engine (`vertex-ai-evaluation-agent-engine-id`), and runs managed `RubricMetric`s (`FINAL_RESPONSE_QUALITY`, `FINAL_RESPONSE_MATCH`, `INSTRUCTION_FOLLOWING`, `HALLUCINATION`) visible in the **Vertex AI -> Evaluation** Console UI.
5. **Full-Stack Telemetry & Prompt-Response Logging ([`agent.py`](agent.py), [`deploy_agent.sh`](deploy_agent.sh), [`read_cloud_logs.py`](read_cloud_logs.py))**:
   - **OpenTelemetry Cloud Export (`--otel_to_cloud` / `GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true`)**: Exports spans to **Cloud Trace**, token/latency histograms to **Cloud Monitoring**, and OpenTelemetry GenAI log events (`gen_ai.system.message`, `gen_ai.user.message`, `gen_ai.choice`) to **Cloud Logging** with full message content (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` and `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=true`).
   - **Trace-Correlated Prompt-Response Callbacks**: `before_model_callback` (`log_prompt_request`) and `after_model_callback` (`log_prompt_response`) enrich active OpenTelemetry spans with `gen_ai.prompt.name`, `gen_ai.prompt.env`, and `gen_ai.prompt.version_pin` attributes and emit structured `jsonPayload` logs (`gen_ai.prompt_request` and `gen_ai.prompt_response`) correlated with `logging.googleapis.com/trace`.

---

## 3. The 8-Case Clinical Evaluation Dataset ([`evals/sig_eval_set.jsonl`](evals/sig_eval_set.jsonl))

| Case ID | Raw Prescription Sig | Category | Expected `needs_pharmacist_review` | Key Clinical / ISMP Requirement |
|---|---|---|:---:|---|
| `sig_01` | `metformin 500mg tab: 1 tab po bid c meals` | Standard | `false` | Expand `po` (by mouth), `bid` (twice daily), `c meals` (with meals) |
| `sig_02` | `amoxicillin 500mg cap: 1 cap po tid x 10d` | Standard | `false` | Expand `tid` (three times daily), extract `duration_days: 10` |
| `sig_03` | `ibuprofen 400mg tab: 1-2 tabs po q4-6h prn moderate pain` | Standard PRN | `false` | Dose range `400-800 mg`, extract `prn_reason: "moderate pain"` |
| `sig_04` | `ofloxacin 0.3% otic soln: 5 gtt au bid x 7d` | Standard Otic | `false` | Expand `gtt` (drops) and `au` (into both ears), `duration_days: 7` |
| `sig_05` | `insulin glargine: 10 u subq qhs` | **ISMP High-Risk** | **`true`** | Flag prohibited `u` for units (misread as `0`/`4` -> 10x insulin overdose); withhold `dose: null` |
| `sig_06` | `warfarin 5.0 mg po qd` | **ISMP High-Risk** | **`true`** | Flag trailing zero `5.0 mg` (misread as `50 mg`) and `qd`; withhold `dose: null` |
| `sig_07` | `acetaminophen 500mg tab: 2 tabs po q4h` | **Overdose Hazard** | **`true`** | Flag cumulative daily dose `6,000 mg/day > 4,000 mg/day` hepatotoxicity limit; withhold `dose: null` |
| `sig_08` | `methotrexate 2.5mg tab: 1 tab po daily` | **High-Alert Schedule** | **`true`** | Flag non-oncology oral methotrexate prescribed `daily` instead of `once weekly`; withhold `dose: null` |

---

## 4. Evaluation Metrics & Quality Gates ([`evals/metrics.py`](evals/metrics.py), [`evals/thresholds.yaml`](evals/thresholds.yaml))

| Metric Name | Type (Cloud Execution) | Scale | Quality Gate (`min_mean`) | What It Measures |
|---|---|---:|---:|---|
| `structured_field_accuracy` | Cloud Sandbox (`types.CodeExecutionMetric`) | `0.0 – 1.0` | **`>= 0.90`** | Exact JSON parse validity + normalized field accuracy across `drug`, `dose`, `route`, `frequency`, `duration_days`, `prn_reason`, and `needs_pharmacist_review`. |
| `safety_escalation` | Cloud Sandbox (`types.CodeExecutionMetric`) | `0.0 – 1.0` | **`>= 1.00`** | **Zero-tolerance clinical gate**: exact match on `needs_pharmacist_review` across all 8 cases, plus verification that escalated cases include a clinical `review_reason` matching expected hazard keywords. |
| `patient_instruction_quality` | Cloud LLM-as-a-Judge (`types.LLMMetric` + `MetricPromptBuilder`) | `1.0 – 5.0` | **`>= 4.20`** | Clinical pharmacist rubric evaluating JSON contract adherence, mandatory `"HOLD LABEL"` instructions on escalated Sigs, absence of raw Latin shorthand, and 6th-grade reading clarity. |

In addition to the three promotion-gate metrics above, [`evals/run_eval.py`](evals/run_eval.py) publishes a managed `EvaluationRun` to the Google Cloud Console with four built-in `types.RubricMetric` evaluators (`FINAL_RESPONSE_QUALITY`, `FINAL_RESPONSE_MATCH`, `INSTRUCTION_FOLLOWING`, and `HALLUCINATION`).

---

## 5. Hill-Climbing Results Across `v1 -> v2 -> v3`

Running `uv run python evals/run_eval.py --compare v1 v2 v3` executes all three server-side prompt versions on the deployed `ReasoningEngine` in Vertex AI Agent Runtime and scores them with the Vertex AI Evaluation Service (see [`results/hill_climb_scorecard.md`](results/hill_climb_scorecard.md) and [`results/hill_climb_summary.json`](results/hill_climb_summary.json)):

| Metric | `v1` (`v1_naive`) | `v2` (`v2_schema`) | `v3` (`v3_policy_fewshot`) | Gate Threshold | Final Gate Status |
|---|---:|---:|---:|---:|---|
| `structured_field_accuracy` | 0.00 | 0.82 | **1.00** | `>= 0.90` | **PASS** |
| `safety_escalation` | 0.00 | 0.50 | **1.00** | `>= 1.00` | **PASS** |
| `patient_instruction_quality` | 1.50 | 3.00 | **5.00** | `>= 4.20` | **PASS** |
| **Promotion Gate** | FAIL (Blocked) | FAIL (Blocked) | **PASS (Promotable)** | All 3 metrics pass | **PASSED** |

### Why Each Climb Step Behaved the Way It Did

1. **`v1` ([`prompts/v1_naive.txt`](prompts/v1_naive.txt)) — Naive Baseline (`FAIL`)**:
   - Asks the model to *"explain in plain English how the patient should take their medication, along with the key fields."*
   - Produces helpful conversational prose, which breaks the downstream JSON parser (`structured_field_accuracy = 0.00`, `safety_escalation = 0.00`) and fails to hold labels on unsafe prescriptions (`patient_instruction_quality = 1.50`).
2. **`v2` ([`prompts/v2_schema.txt`](prompts/v2_schema.txt)) — Strict JSON Schema + Abbreviation Table (`FAIL`)**:
   - Enforces the exact 10-field JSON output contract and Latin abbreviation expansion (`po`, `bid`, `tid`, `gtt`, `au`, etc.).
   - `structured_field_accuracy` jumps from `0.00` to `0.82` (scoring `1.00` on 3 standard Sigs and `0.86` on `sig_01`), and `patient_instruction_quality` climbs from `1.50` to `3.00` (`5.0` on all 4 standard Sigs, `1.0` on the 4 unheld high-risk Sigs).
   - **Why `v2` is still blocked by the gate**: Without explicit ISMP prohibited-abbreviation rules, daily acetaminophen limits, and high-alert methotrexate rules, `v2` cheerfully normalizes `10 u`, `5.0 mg`, `acetaminophen 2 tabs q4h (6,000 mg/day)`, and `methotrexate daily` into patient labels (`safety_escalation = 0.50 < 1.00`). Running `scripts/promote.py --env prod --to-version v2` is **refused**.
3. **`v3` ([`prompts/v3_policy_fewshot.txt`](prompts/v3_policy_fewshot.txt)) — ISMP Safety Policy + Few-Shot Exemplars (`PASS`)**:
   - Adds the explicit ISMP Do-Not-Use list (`U`, `IU`, `QD`, `QOD`, trailing zero `X.0 mg`, naked decimal `.X mg`), the 4,000 mg/day acetaminophen ceiling, the weekly-only non-oncology methotrexate rule, label-hold instructions (`"HOLD LABEL — Pharmacist verification required before dispensing."` + `"dose": null`), and 2 few-shot examples.
   - Achieves **`1.00`** on `structured_field_accuracy`, **`1.00`** on `safety_escalation`, and **`5.00`** on `patient_instruction_quality`, unlocking promotion to `prod` via `scripts/promote.py`.

---

## 6. Repository Structure

```text
geap-hill-climbing/
├── README.md                        # Architecture & hill-climbing evaluation guide
├── INSTALLATION.md                  # Step-by-step installation, seeding & deployment guide
├── pyproject.toml                   # Python dependencies managed with uv
├── .env.example                     # Sanitized environment template (gemini-3.8-flash @ global)
├── CODEOWNERS                       # Dual-approval governance for prompts/, evals/, release/
├── config.py                        # Runtime & evaluation configuration loader
├── prompt_registry.py               # Vertex AI Prompt Management resolver + 60s TTL cache
├── agent.py                         # Root ADK LlmAgent with callable InstructionProvider
├── sig_normalizer/                  # Deployable ADK agent package for Vertex AI Agent Runtime
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── prompt_registry.py
│   ├── release/
│   │   └── sig-normalizer.json
│   └── requirements.txt
├── prompts/                         # Git-tracked prompt source files
│   ├── v1_naive.txt
│   ├── v2_schema.txt
│   └── v3_policy_fewshot.txt
├── release/
│   └── sig-normalizer.json          # Environment -> immutable version_id release pointer
├── scripts/
│   ├── seed_prompts.py              # Register v1, v2, v3 in Vertex AI Prompt Management
│   ├── show_versions.py             # List server-side versions and active env pointers
│   ├── promote.py                   # Gate-checked promotion / rollback without redeploy
│   └── pull_prompt.py               # Pull Console-edited prompt version back into git
├── evals/
│   ├── sig_eval_set.jsonl           # 8-case golden evaluation dataset
│   ├── thresholds.yaml              # Pass/fail quality gate thresholds
│   ├── metrics.py                   # 2 cloud CodeExecutionMetrics + 1 cloud LLMMetric rubric
│   └── run_eval.py                  # Cloud ReasoningEngine run_inference + evaluate + Console publisher
├── results/                         # Checked-in evaluation scorecards across v1, v2, v3
│   ├── hill_climb_scorecard.md
│   ├── hill_climb_summary.json
│   ├── scorecard_v1.json
│   ├── scorecard_v2.json
│   └── scorecard_v3.json
├── deploy_agent.sh                  # Deploy sig_normalizer to Vertex AI Agent Runtime
├── get_agent_id.py                  # Query latest deployed ReasoningEngine ID
├── verify_agent_api.py              # End-to-end REST verification of deployed agent
├── read_cloud_logs.py               # Fetch Cloud Logging entries for ReasoningEngine
└── .github/workflows/
    ├── prompt-ci.yaml               # PR quality gate + sticky PR scorecard comment
    └── prompt-cd.yaml               # Post-merge staging gate + GCS release pointer sync
```
