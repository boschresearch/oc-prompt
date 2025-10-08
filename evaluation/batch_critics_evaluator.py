#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys

# =============================================================================
# Ensure project root is on the Python module search path
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import glob
import math
import logging
import time
import random
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# -----------------------------------------------------------------------------
# LangChain model backends
# -----------------------------------------------------------------------------
from langchain_openai import AzureChatOpenAI
try:
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
except Exception:
    AzureAIChatCompletionsModel = None  # optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from util import sort_on_before_underscore

# =============================================================================
# Helper to resolve paths against PROJECT_ROOT
# =============================================================================
def resolve_path(path: str) -> str:
    """If `path` is relative, prepend PROJECT_ROOT."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(description="Run recipe evaluation with configurable profile")
parser.add_argument(
    "--profile",
    type=str,
    default=resolve_path("data/recipe/config/evaluation.json"),
    help="Path to JSON profile (default: data/recipe/config/evaluation.json)"
)
parser.add_argument("--max-retries", type=int, default=5, help="Max retries per recipe on transient/model-output errors")
parser.add_argument("--base-backoff-seconds", type=float, default=1.0, help="Base seconds for exponential backoff")
parser.add_argument(
    "--output-subdir",
    type=str,
    default=None,
    help=(
        "Re-use a specific output subdirectory under evaluation_output_dir instead of creating a new timestamped one. "
        "Example: --output-subdir evaluation_20250310_134552"
    ),
)
args = parser.parse_args()

# =============================================================================
# Load profile
# =============================================================================
with open(args.profile, 'r', encoding='utf-8') as f:
    profile = json.load(f)

translation_data_dir  = resolve_path(profile["translation_data_dir"])
target_data_dir       = resolve_path(profile["target_data_dir"])
evaluation_output_dir = resolve_path(profile["evaluation_output_dir"])
model_config_file     = resolve_path(profile["model_config_file"])

test_approach_list    = profile["test_approach_list"]
test_llm_list         = profile["test_llm_list"]
evaluation_steps      = profile["evaluation_steps"]
execution_mode        = str(profile.get("execution_mode", "target")).strip().lower()  # "target" | "source"

def _coerce_max_recipes(val: Any) -> Optional[int]:
    """Accept int-like values. None/''/invalid/<=0 -> None (evaluate all)."""
    if val is None:
        return None
    if isinstance(val, int):
        return val if val > 0 else None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if s.isdigit():
            v = int(s);  return v if v > 0 else None
    return None

max_recipes: Optional[int] = _coerce_max_recipes(profile.get("max_recipes"))

instruction       = profile["instruction"]
metric_name       = profile["metric_name"]
json_src_text     = profile["json_fields"]["source"]["recipe_text"]
json_src_name     = profile["json_fields"]["source"]["recipe_name"]
json_tgt_text     = profile["json_fields"]["target"]["recipe_text"]
json_tgt_name     = profile["json_fields"]["target"]["recipe_name"]
output_id_col     = profile["output_fields"]["id"]
output_name_col   = profile["output_fields"]["name"]
combine_template  = profile["combine_template"]

# =============================================================================
# Logging
# =============================================================================
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(target_data_dir, exist_ok=True)
log_file = os.path.join(target_data_dir, f"batch_critics_evaluator_{ts}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # adjust to INFO if needed
ch = logging.StreamHandler()
fh = logging.FileHandler(log_file, encoding="utf-8")
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(fmt); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)

logger.info("=== Recipe Evaluation Runner started ===")
logger.info(f"Profile: {args.profile}")
logger.info(f"Execution mode: {execution_mode} (target|source)")
logger.info(f"Translation dir: {translation_data_dir}")
logger.info(f"Target dir: {target_data_dir}")
logger.info(f"Output dir (root): {evaluation_output_dir}")
logger.info(f"Model config file: {model_config_file}")
logger.info(f"max_recipes: {max_recipes if max_recipes is not None else 'no limit (evaluate all)'}")
logger.info(f"retry policy: max_retries={args.max_retries}, base_backoff_seconds={args.base_backoff_seconds}")
logger.info(f"output-subdir override: {args.output_subdir if args.output_subdir else '(none -> use timestamp)'}")

# =============================================================================
# DeepEval adapter — shape compatible with deepeval 3.4.8
# =============================================================================
class _ShimMessage:
    """Mimic OpenAI message structure."""
    def __init__(self, content: str):
        self.content = content

class _ShimChoice:
    """Mimic OpenAI choice structure."""
    def __init__(self, content: str):
        self.message = _ShimMessage(content)

class RawResponseShim(str):
    """
    String-like response that also exposes:
      - .choices[0].message.content  (OpenAI-style)
      - .score and .reason           (fallback path; left as None/'')
    This lets deepeval treat the same object as raw text or as a chat-like response.
    """
    def __new__(cls, text: str, cost: float = 0.0, score: Optional[float] = None, reason: str = ""):
        obj = super().__new__(cls, text)
        obj._text = text
        obj._cost = float(cost)
        obj.score = score
        obj.reason = reason
        obj.choices = [_ShimChoice(text)]
        return obj
    @property
    def cost(self) -> float:
        return self._cost
    def __repr__(self):
        return f"RawResponseShim(len={len(self._text)}, cost={self._cost})"

class EvaluationLLM(DeepEvalBaseLLM):
    """
    Adapter for deepeval 3.4.8:
    - generate()/a_generate(): return plain text (string).
    - generate_raw_response()/a_generate_raw_response():
        return **(RawResponseShim, 0.0)** tuple  ← IMPORTANT
    - Do NOT forward unknown kwargs ('schema', 'top_logprobs', ...) to the LC model.
    """
    def __init__(self, model, name: str = "Evaluation LLM"):
        self.model = model
        self._name = name

    def load_model(self):
        return self.model

    # ---- sync paths
    def generate(self, prompt: str, **kwargs) -> str:
        return self.model.invoke(prompt).content

    def generate_raw_response(self, prompt: str, **kwargs) -> Tuple[RawResponseShim, float]:
        text = self.generate(prompt)
        shim = RawResponseShim(text=text, cost=0.0)
        return shim, 0.0

    # ---- async paths
    async def a_generate(self, prompt: str, **kwargs) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    async def a_generate_raw_response(self, prompt: str, **kwargs) -> Tuple[RawResponseShim, float]:
        text = await self.a_generate(prompt)
        shim = RawResponseShim(text=text, cost=0.0)
        return shim, 0.0

    def get_model_name(self) -> str:
        return self._name

# =============================================================================
# Env loader
# =============================================================================
def load_env_variables(fp: str) -> Dict[str, str]:
    """Load KEY=VALUE pairs (lines starting with '#' ignored)."""
    env: Dict[str, str] = {}
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

# =============================================================================
# Unified evaluator model configurator (DeepSeek/Azure auto-detect)
# =============================================================================
def configure_evaluator_model(fp: str) -> EvaluationLLM:
    """
    Azure OpenAI expects:
      AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
    Azure AI Inference (DeepSeek) expects:
      AZURE_INFERENCE_CREDENTIAL, AZURE_INFERENCE_ENDPOINT, MODEL_NAME, API_VERSION
    """
    logger.info(f"Loading evaluator env from: {fp}")
    env = load_env_variables(fp)
    logger.debug(f"Loaded keys: {list(env.keys())}")

    has_azure_openai = all(k in env for k in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ])
    has_azure_inference = all(k in env for k in [
        "AZURE_INFERENCE_CREDENTIAL",
        "AZURE_INFERENCE_ENDPOINT",
        "MODEL_NAME",
        "API_VERSION",
    ])

    if has_azure_openai:
        logger.info("Using Azure OpenAI backend (AzureChatOpenAI).")
        os.environ.update({
            "AZURE_OPENAI_API_KEY":     env.get("AZURE_OPENAI_API_KEY", ""),
            "AZURE_OPENAI_ENDPOINT":    env.get("AZURE_OPENAI_ENDPOINT", ""),
            "AZURE_OPENAI_DEPLOYMENT":  env.get("AZURE_OPENAI_DEPLOYMENT", ""),
            "AZURE_OPENAI_API_VERSION": env.get("AZURE_OPENAI_API_VERSION", ""),
        })
        chat = AzureChatOpenAI(
            azure_endpoint=env.get("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=env.get("AZURE_OPENAI_DEPLOYMENT"),
            api_version=env.get("AZURE_OPENAI_API_VERSION"),
            temperature=0,
        )
        return EvaluationLLM(model=chat, name="Azure OpenAI Chat (evaluation)")

    if has_azure_inference:
        logger.info("Using Azure AI Inference backend (DeepSeek via Azure).")
        if AzureAIChatCompletionsModel is None:
            raise RuntimeError(
                "langchain-azure-ai is not installed, but Azure AI Inference "
                "(DeepSeek) config was provided. Please install it first:\n"
                "  pip install langchain-azure-ai"
            )
        chat = AzureAIChatCompletionsModel(
            credential=env.get("AZURE_INFERENCE_CREDENTIAL"),
            endpoint=env.get("AZURE_INFERENCE_ENDPOINT"),
            model=env.get("MODEL_NAME"),
            api_version=env.get("API_VERSION"),
            temperature=0,
        )
        return EvaluationLLM(model=chat, name="Azure AI Inference (DeepSeek)")

    raise ValueError(
        "Unrecognized evaluator config file. Expected either Azure OpenAI keys "
        "or Azure AI Inference (DeepSeek) keys."
    )

# =============================================================================
# Statistics helper
# =============================================================================
def compute_statistics(vals, conf=0.95):
    n = len(vals)
    if n == 0:
        return None
    arr = np.array(vals)
    m = arr.mean()
    mn = arr.min()
    mx = arr.max()
    sd = arr.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        se = sd / math.sqrt(n)
        t = stats.t.ppf((1 + conf) / 2.0, n - 1)
        margin = t * se
        ci = (m - margin, m + margin)
    else:
        ci = (m, m)
    return {"count": n, "mean": m, "min": mn, "max": mx, "std": sd, "conf_interval": ci}

# =============================================================================
# Metric setup
# =============================================================================
def setup_critics_metric(model, steps) -> GEval:
    logger.info(f"Setting up GEval metric: {metric_name}")
    metric = GEval(
        name=metric_name,
        evaluation_steps=steps,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        threshold=0.5,     # GEval's internal scale is 0–1
        strict_mode=False,
        verbose_mode=False,
    )
    # Some deepeval codepaths do "+= cost"
    try:
        if not hasattr(metric, "evaluation_cost") or metric.evaluation_cost is None:
            metric.evaluation_cost = 0.0
            logger.debug("Initialized critics_metric.evaluation_cost=0.0")
    except Exception:
        pass
    return metric

# =============================================================================
# Data loaders (target + source) with optional max_recipes limit
# =============================================================================
def read_target_recipe_data(
    base: str,
    approach: str,
    llm: str,
    limit: Optional[int] = None
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict[str, dict]]:
    """Read candidate TARGET recipes; apply limit if provided."""
    logger.info(f"[Load targets] approach={approach} llm={llm} limit={limit if limit else 'no limit'}")
    cand, names, usage = {}, {}, {}
    dirp = os.path.join(base, approach, llm)
    if not os.path.isdir(dirp):
        logger.warning(f"No target dir: {dirp}")
        return cand, names, usage

    rids_all = sorted(os.listdir(dirp), key=sort_on_before_underscore)
    rids = rids_all[:limit] if (isinstance(limit, int) and limit > 0) else rids_all
    logger.info(f"Found {len(rids_all)} recipe IDs under {dirp}; selected {len(rids)} for evaluation.")

    for rid in rids:
        folder = os.path.join(dirp, rid)
        files = glob.glob(os.path.join(folder, "*.json"))
        tot_toks = tot_cost = tot_time = 0.0
        txt, nm = None, None
        for jf in files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                txt = txt or d.get(json_tgt_text)
                nm  = nm  or d.get(json_tgt_name)
                tu  = d.get("token_usage", {})
                tot_toks += tu.get("total_tokens", 0)
                tot_cost += tu.get("total_cost", 0)
                tot_time += d.get("time_taken_seconds", 0)
            except Exception as e:
                logger.error(f"Error reading {jf}: {e}")
        cand[rid] = txt
        names[rid] = nm
        usage[rid] = {
            "total_tokens": tot_toks,
            "total_cost": tot_cost,
            "time_taken_seconds": tot_time,
        }
    logger.debug(f"Loaded target entries: {len(cand)} (approach={approach}, llm={llm})")
    return cand, names, usage

def read_source_recipes(
    limit: Optional[int] = None
) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]], Dict[str, dict]]:
    """
    Read SOURCE (original/translation) recipes from translation_data_dir.
    Returns id->text, id->name, id->usage (zeros), honoring 'limit'.
    Uses filename (without extension) as recipe id to keep it stable across runs.
    """
    logger.info(f"[Load source] dir={translation_data_dir} limit={limit if limit else 'no limit'}")
    cand, names, usage = {}, {}, {}

    if not os.path.isdir(translation_data_dir):
        logger.warning(f"Translation dir not found: {translation_data_dir}")
        return cand, names, usage

    files_all = [fn for fn in os.listdir(translation_data_dir) if fn.endswith(".json")]
    files_sorted = sorted(files_all, key=sort_on_before_underscore)
    files = files_sorted[:limit] if (isinstance(limit, int) and limit > 0) else files_sorted
    logger.info(f"Found {len(files_all)} source JSON files; selected {len(files)} for evaluation.")

    for fn in files:
        rid = os.path.splitext(fn)[0]  # stable id
        path = os.path.join(translation_data_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            cand[rid] = d.get(json_src_text)
            names[rid] = d.get(json_src_name)
            usage[rid] = {
                "total_tokens": 0.0,
                "total_cost": 0.0,
                "time_taken_seconds": 0.0,
            }
        except Exception as e:
            logger.error(f"Failed to read source file {path}: {e}")

    logger.debug(f"Loaded source entries: {len(cand)}")
    return cand, names, usage

# =============================================================================
# Retry helpers + checkpointing
# =============================================================================
def _scale_to_0_10(x: float) -> float:
    """Convert GEval's 0–1 score to a 0–10 scale, clamped."""
    if x is None:
        return 0.0
    y = float(x) * 10.0
    return max(0.0, min(10.0, y))

def _should_retry(exc: Exception) -> bool:
    """
    Retry on:
    - network timeouts / transient transport errors
    - deepeval JSON parsing errors
    - kwargs/contract mismatches ('schema', 'top_logprobs')
    - return-shape/cost arithmetic / attribute expectations
    """
    low = f"{type(exc).__name__}: {exc}".lower()
    patterns = [
        # network / transport
        "timeout", "timed out", "readtimeout", "connecttimeout",
        "serviceresponsetimeouterror", "sockettimeouterror",
        # model output / JSON format
        "invalid json", "invalid control character", "jsondecodeerror",
        # API contract / kwargs mismatches
        "unexpected keyword argument 'schema'", "a_generate_raw_response", "top_logprobs",
        # return-shape / cost arithmetic / attribute expectations
        "too many values to unpack", "evaluation_cost", "unsupported operand type(s) for +=",
        "object has no attribute 'score'", "object has no attribute 'choices'",
    ]
    return any(p in low for p in patterns)

def _exp_backoff_sleep(attempt: int, base: float):
    sleep_s = base * (2 ** (attempt - 1))
    sleep_s += random.random() * 0.5  # jitter
    time.sleep(sleep_s)

def _checkpoint_path_for(out_xlsx: str) -> str:
    base, _ = os.path.splitext(out_xlsx)
    return base + ".checkpoint.jsonl"

def _failed_path_for(out_xlsx: str) -> str:
    base, _ = os.path.splitext(out_xlsx)
    return base + ".failed.jsonl"

def _load_checkpoint(path: str) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = str(rec.get(output_id_col))
                if rid:
                    done[rid] = rec
            except Exception:
                continue
    return done

def _append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =============================================================================
# Evaluation + Excel export (with retries & resume)
# =============================================================================
def run_evaluation(metric, cand, names, usage, out_xlsx):
    logger.info(f"Running evaluation for {len(cand)} recipes...")
    ckpt_path   = _checkpoint_path_for(out_xlsx)
    failed_path = _failed_path_for(out_xlsx)
    done_map    = _load_checkpoint(ckpt_path)
    done_ids    = set(done_map.keys())

    all_ids     = list(cand.keys())
    pending_ids = [rid for rid in all_ids if rid not in done_ids]

    logger.info(f"Checkpoint: {len(done_ids)} already done, {len(pending_ids)} pending.")
    pbar = tqdm(total=len(all_ids), initial=len(done_ids), desc="Evaluating recipes", dynamic_ncols=True)

    new_records: List[dict] = []

    for rid in pending_ids:
        txt = cand[rid]
        nm  = names.get(rid, "")
        logger.info(f"Evaluating recipe_id={rid} recipe_name={nm}")

        combined = combine_template.format(id=rid, name=nm, text=txt)
        tc = LLMTestCase(name=rid, input=instruction, actual_output=combined)

        attempt = 0
        while True:
            try:
                attempt += 1
                metric.measure(tc)  # may raise
                sc_raw = float(metric.score)       # 0–1
                sc_10  = _scale_to_0_10(sc_raw)    # 0–10 for reporting
                expl   = metric.reason

                rec = {
                    output_id_col: rid,
                    output_name_col: nm,
                    "score_raw_0_1": sc_raw,
                    "score_0_10": sc_10,
                    "explanation": expl
                }
                _append_jsonl(ckpt_path, rec)
                new_records.append(rec)
                logger.debug(f"Scores for {rid}: raw_0_1={sc_raw:.4f} scaled_0_10={sc_10:.2f}")
                break

            except Exception as e:
                if attempt <= args.max_retries and _should_retry(e):
                    logger.warning(f"[retry {attempt}/{args.max_retries}] error on {rid}: {e}")
                    _exp_backoff_sleep(attempt, args.base_backoff_seconds)
                    # defensive: ensure deepeval's internal cost accumulator is numeric
                    try:
                        if not hasattr(metric, "evaluation_cost") or metric.evaluation_cost is None:
                            setattr(metric, "evaluation_cost", 0.0)
                            logger.debug("Initialized metric.evaluation_cost=0.0 for safety")
                    except Exception:
                        pass
                    continue

                fail_rec = {
                    output_id_col: rid,
                    output_name_col: nm,
                    "error": f"{type(e).__name__}: {e}",
                    "attempts": attempt
                }
                _append_jsonl(failed_path, fail_rec)
                logger.error(f"Evaluation failed for {rid} after {attempt} attempts; aborting batch.")
                raise

        pbar.update(1)

    pbar.close()

    # Combine checkpointed + newly evaluated
    all_records = list(done_map.values()) + new_records
    df_scores = pd.DataFrame(all_records)

    # summary stats on 0–10 scale
    if not df_scores.empty:
        s = df_scores["score_0_10"]
        summary = {
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=1),
            "count": len(s),
        }
        if summary["count"] > 1:
            se = summary["std"] / math.sqrt(summary["count"])
            ci = 1.96 * se
            summary["ci_lower"] = summary["mean"] - ci
            summary["ci_upper"] = summary["mean"] + ci
        else:
            summary["ci_lower"] = summary["ci_upper"] = None
        df_summary = pd.DataFrame([summary])
    else:
        df_summary = pd.DataFrame([{
            "min": None, "max": None, "mean": None, "median": None,
            "std": None, "count": 0, "ci_lower": None, "ci_upper": None
        }])

    # token stats
    raw = [{output_id_col: rid, **u} for rid, u in usage.items()]
    df_raw = pd.DataFrame(raw)

    def _safe_stats(values):
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "std": 0, "conf_interval": (0, 0)}
        return compute_statistics(values)

    tok_stats  = _safe_stats([u.get("total_tokens", 0) for u in usage.values()])
    cost_stats = _safe_stats([u.get("total_cost", 0) for u in usage.values()])
    time_stats = _safe_stats([u.get("time_taken_seconds", 0) for u in usage.values()])

    df_tokstat = pd.DataFrame({
        "metric": ["total_tokens", "total_cost", "time_taken_seconds"],
        "count":  [tok_stats["count"],  cost_stats["count"],  time_stats["count"]],
        "mean":   [tok_stats["mean"],   cost_stats["mean"],   time_stats["mean"]],
        "min":    [tok_stats["min"],    cost_stats["min"],    time_stats["min"]],
        "max":    [tok_stats["max"],    cost_stats["max"],    time_stats["max"]],
        "std":    [tok_stats["std"],    cost_stats["std"],    time_stats["std"]],
        "ci_lower":[tok_stats["conf_interval"][0], cost_stats["conf_interval"][0], time_stats["conf_interval"][0]],
        "ci_upper":[tok_stats["conf_interval"][1], cost_stats["conf_interval"][1], time_stats["conf_interval"][1]],
    })

    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_scores.to_excel(w, sheet_name="scores", index=False)
        df_summary.to_excel(w, sheet_name="stats_0_10",  index=False)
        df_raw.to_excel(w, sheet_name="token_usage_raw", index=False)
        df_tokstat.to_excel(w, sheet_name="token_usage_stats", index=False)

    logger.info(f"Saved evaluation workbook: {out_xlsx}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Failures (if any) recorded at: {failed_path}")

# =============================================================================
# Batch runners
# =============================================================================
def _resolve_output_root(base_out_dir: str, subdir_arg: Optional[str]) -> str:
    """
    Decide where to put outputs:
      - If --output-subdir is provided, reuse that folder under base_out_dir.
      - Else, create a new timestamped folder under base_out_dir.
    """
    if subdir_arg:
        root = os.path.join(base_out_dir, subdir_arg)
        os.makedirs(root, exist_ok=True)
        return root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(base_out_dir, f"evaluation_{ts}")
    os.makedirs(root, exist_ok=True)
    return root

def process_target_recipe_directory(
    metric,
    base_dir: str,
    approaches,
    llms,
    out_dir: str,
    limit: Optional[int] = None,
    output_subdir: Optional[str] = None,
):
    root = _resolve_output_root(out_dir, output_subdir)
    logger.info(f"Batch output root: {root}  (persisted across runs if --output-subdir is used)")
    logger.info(f"Per-LLM recipe cap (max_recipes): {limit if limit else 'no limit'}")

    for app in approaches:
        for llm in llms:
            logger.info(f"=== Start batch: approach={app} | llm={llm} ===")
            sub = os.path.join(root, app, llm)
            os.makedirs(sub, exist_ok=True)
            cand, names, usage = read_target_recipe_data(base_dir, app, llm, limit=limit)
            if not cand:
                logger.info(f"Skip (no candidates) for {app}/{llm}")
                continue
            fname = f"{app}_{llm}_recipe_evaluation.xlsx"
            out_path = os.path.join(sub, fname)
            run_evaluation(metric, cand, names, usage, out_path)
            logger.info(f"=== Finished batch: approach={app} | llm={llm} ===")

def process_source_recipe_directory(
    metric,
    translation_dir: str,
    out_dir: str,
    limit: Optional[int] = None,
    output_subdir: Optional[str] = None,
):
    """
    Evaluate SOURCE recipes (originals) under translation_dir.
    Results are saved under the same evaluation root pattern as TARGET mode:
      <evaluation_output_dir>/<evaluation_{timestamp} or --output-subdir>/source/source_recipe_evaluation.xlsx
    """
    root = _resolve_output_root(out_dir, output_subdir)
    logger.info(f"Batch output root: {root}  (persisted across runs if --output-subdir is used)")
    logger.info(f"Per-LLM recipe cap (max_recipes): {limit if limit else 'no limit'}")

    sub = os.path.join(root, "source")  # keep structure consistent under the evaluation root
    os.makedirs(sub, exist_ok=True)

    cand, names, usage = read_source_recipes(limit=limit)
    if not cand:
        logger.info("Skip (no source candidates found)")
        return

    fname = "source_recipe_evaluation.xlsx"
    out_path = os.path.join(sub, fname)
    run_evaluation(metric, cand, names, usage, out_path)
    logger.info("=== Finished batch: source ===")

# =============================================================================
# main()
# =============================================================================
def main():
    logger.info("Configuring evaluator model...")
    eval_llm = configure_evaluator_model(model_config_file)
    logger.info(f"Evaluator model ready: {eval_llm.get_model_name()}")

    critics_metric = setup_critics_metric(eval_llm, evaluation_steps)

    logger.info("Starting batch processing...")
    if execution_mode == "target":
        process_target_recipe_directory(
            critics_metric,
            target_data_dir,
            test_approach_list,
            test_llm_list,
            evaluation_output_dir,
            limit=max_recipes,
            output_subdir=args.output_subdir,
        )
    elif execution_mode == "source":
        process_source_recipe_directory(
            critics_metric,
            translation_data_dir,
            evaluation_output_dir,
            limit=max_recipes,
            output_subdir=args.output_subdir,
        )
    else:
        raise ValueError(f"Unsupported execution_mode: {execution_mode} (expected 'target' or 'source')")

    logger.info("All done ✅")

if __name__ == "__main__":
    main()
