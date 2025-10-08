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

"""
RCF Reliability & Validity Evaluation Script
- Location: evaluation/rcf_reliability.py
- Input CSV: data/recipe/evaluation/rcf_input.csv
- Required columns: recipe_id | expert1 | expert2 | expert3 | rcf (e.g., 'rcf from LLM' also accepted)
- Stats: Panel reliability (ICC(2,k), Cronbach's alpha), rank correlations (Spearman/Kendall) with bootstrap CIs,
         per-rater correlations + Fisher's z-mean, calibration regression (panel_mean ~ RCF)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, zscore
import pingouin as pg
import statsmodels.api as sm


# --------------------------
# Logging configuration
# --------------------------
LOGGER_NAME = "rcf_eval"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)


# --------------------------
# Dataclasses for results
# --------------------------
@dataclass
class CorrResult:
    target: str
    spearman_rho: float
    spearman_ci: Tuple[float, float]
    spearman_p: float
    kendall_tau_b: float
    kendall_ci: Tuple[float, float]
    kendall_p: float


# --------------------------
# Utilities
# --------------------------
def numeric_summary(df: pd.DataFrame) -> str:
    """
    Version-agnostic numeric-only summary for pandas 1.x/2.x.
    Selects numeric columns and calls .describe() without numeric_only kw.
    """
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return "No numeric columns found."
    # 보기 좋게 소수점 정리
    with pd.option_context("display.float_format", lambda v: f"{v:0.3f}"):
        return num.describe().to_string()

def _validate_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures required columns are present. Strips whitespace and normalizes names.
    Accepts common variants of the 'rcf' column name (e.g., 'rcf from LLM' -> 'rcf').
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = ["recipe_id", "expert1", "expert2", "expert3"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in CSV.")

    # Detect/standardize rcf column
    rcf_candidates = [c for c in df.columns if c == "rcf" or c.startswith("rcf")]
    if not rcf_candidates:
        raise ValueError(
            "Missing 'rcf' column. Acceptable names include 'rcf' or variants starting with 'rcf' "
            "(e.g., 'rcf_from_llm')."
        )
    if "rcf" in rcf_candidates:
        rcf_col = "rcf"
    else:
        rcf_col = rcf_candidates[0]
        logger.warning("Using '%s' as RCF column (standardized to 'rcf').", rcf_col)
        df = df.rename(columns={rcf_col: "rcf"})

    # Type checks
    numeric_cols = ["expert1", "expert2", "expert3", "rcf"]
    for col in numeric_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception as e:
                raise ValueError(f"Column '{col}' must be numeric. Error: {e}") from e

    # Drop rows with missing scores
    n_before = len(df)
    df = df.dropna(subset=["expert1", "expert2", "expert3", "rcf"])
    n_after = len(df)
    if n_after < n_before:
        logger.warning("Dropped %d rows due to missing values in required score columns.", n_before - n_after)

    return df


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 10000,
    alpha: float = 0.05,
    random_state: int = 2025,
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap CI for a correlation-type statistic.
    """
    rng = np.random.default_rng(random_state)
    n = len(x)
    stats = np.empty(B, dtype=float)
    idxs = rng.integers(0, n, size=(B, n))
    for b in range(B):
        i = idxs[b]
        stats[b] = stat_fn(x[i], y[i])
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def spearman_value(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).correlation)


def kendall_value(x: np.ndarray, y: np.ndarray) -> float:
    # Kendall tau-b; handles ties
    return float(kendalltau(x, y, variant="b").correlation)


def fisher_z_mean(corrs: List[float]) -> float:
    """Fisher's z-transform mean back to correlation."""
    zs = [np.arctanh(c) for c in corrs if np.isfinite(c)]
    if not zs:
        return np.nan
    return float(np.tanh(np.mean(zs)))


# --------------------------
# Core computations
# --------------------------
def build_panel_scores(df: pd.DataFrame, expert_cols: List[str]) -> pd.DataFrame:
    """
    Returns df with panel_mean / panel_median / panel_zmean columns.
    """
    out = df.copy()
    out["panel_mean"] = out[expert_cols].mean(axis=1)
    out["panel_median"] = out[expert_cols].median(axis=1)
    Z = out[expert_cols].apply(zscore)  # rater-wise standardization
    out["panel_zmean"] = Z.mean(axis=1)
    return out


def panel_reliability(df: pd.DataFrame, expert_cols: List[str]) -> Dict[str, object]:
    """
    Computes ICC(2,k) and Cronbach's alpha for panel consistency.
    """
    # Prepare long format for ICC
    long_df = df.melt(
        id_vars=["recipe_id"],
        value_vars=expert_cols,
        var_name="rater",
        value_name="score",
    ).rename(columns={"recipe_id": "target"})

    icc_tbl = pg.intraclass_corr(
        data=long_df,
        targets="target",   # individual recipe
        raters="rater",     # expert id
        ratings="score"
    )
    icc2k_row = icc_tbl.loc[icc_tbl["Type"] == "ICC2k"].copy()
    if icc2k_row.empty:
        raise RuntimeError("ICC2k not computed by pingouin; check inputs.")
    icc2k = float(icc2k_row["ICC"].iloc[0])
    # CI95% in pingouin is a tuple stored per-row; normalize to tuple[float, float]
    icc2k_ci_raw = icc2k_row["CI95%"].iloc[0]
    icc2k_ci = (float(icc2k_ci_raw[0]), float(icc2k_ci_raw[1]))

    alpha_val, _ = pg.cronbach_alpha(data=df[expert_cols])

    return {
        "icc2k": icc2k,
        "icc2k_ci": icc2k_ci,
        "alpha": float(alpha_val),
        "icc_table": icc_tbl
    }


def correlation_suite(
    df: pd.DataFrame,
    targets: List[str],
    x_col: str = "rcf",
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 2025,
) -> pd.DataFrame:
    """
    Computes Spearman rho & Kendall tau-b + bootstrap CIs between RCF and each target.
    """
    results: List[CorrResult] = []
    x = df[x_col].to_numpy()

    for t in targets:
        y = df[t].to_numpy()
        rho, p_rho = spearmanr(x, y)
        tau, p_tau = kendalltau(x, y, variant="b")

        rho_ci = bootstrap_ci(x, y, spearman_value, B=B, alpha=alpha, random_state=seed)
        tau_ci = bootstrap_ci(x, y, kendall_value, B=B, alpha=alpha, random_state=seed)

        results.append(
            CorrResult(
                target=t,
                spearman_rho=float(rho),
                spearman_ci=rho_ci,
                spearman_p=float(p_rho),
                kendall_tau_b=float(tau),
                kendall_ci=tau_ci,
                kendall_p=float(p_tau),
            )
        )

    # Pretty table
    rows: List[Dict[str, str]] = []
    for r in results:
        rows.append(
            {
                "Target": r.target,
                "Spearman ρ": f"{r.spearman_rho: .3f}",
                "ρ 95% CI": f"[{r.spearman_ci[0]:.3f}, {r.spearman_ci[1]:.3f}]",
                "ρ p": f"{r.spearman_p:.3g}",
                "Kendall τ-b": f"{r.kendall_tau_b: .3f}",
                "τ 95% CI": f"[{r.kendall_ci[0]:.3f}, {r.kendall_ci[1]:.3f}]",
                "τ p": f"{r.kendall_p:.3g}",
            }
        )
    return pd.DataFrame(rows)


def per_rater_correlations(
    df: pd.DataFrame,
    expert_cols: List[str],
    x_col: str = "rcf",
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 2025,
) -> Tuple[pd.DataFrame, float]:
    """
    Correlates RCF with each expert individually (Spearman), with bootstrap CIs,
    and computes Fisher's z-mean correlation across experts.
    """
    rows: List[Dict[str, str]] = []
    rhos: List[float] = []
    for e in expert_cols:
        x = df[x_col].to_numpy()
        y = df[e].to_numpy()
        rho, p = spearmanr(x, y)
        ci = bootstrap_ci(x, y, spearman_value, B=B, alpha=alpha, random_state=seed)
        rows.append(
            {
                "Expert": e,
                "Spearman ρ": f"{rho: .3f}",
                "ρ 95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                "ρ p": f"{p:.3g}",
            }
        )
        rhos.append(float(rho))

    rho_meta = fisher_z_mean(rhos)
    tbl = pd.DataFrame(rows)
    return tbl, rho_meta


def calibration_regression(df: pd.DataFrame, y_col: str = "panel_mean", x_col: str = "rcf") -> Dict[str, object]:
    """
    OLS y ~ x; reports slope, intercept, CIs, and R^2.
    - X를 DataFrame으로 유지해 conf_int()가 DataFrame을 반환하도록 유도
    - 혹시 환경에 따라 ndarray가 오더라도 안전하게 처리
    """
    # X를 DataFrame으로 유지 (열 이름 보존: 'const', x_col)
    X = sm.add_constant(df[[x_col]])        # <-- 핵심 수정: DataFrame 유지
    y = df[y_col]                            # Series 그대로 사용

    model = sm.OLS(y, X).fit()
    params = model.params                    # pandas Series (index: 'const', x_col)
    conf = model.conf_int()                  # 보통 DataFrame 반환, 일부 환경에서 ndarray일 수도 있음

    # 파라미터 추정치
    intercept = float(params["const"])
    slope = float(params[x_col])

    # 신뢰구간: DataFrame/ndarray 모두 지원
    if isinstance(conf, pd.DataFrame):
        intercept_ci = (float(conf.loc["const", 0]), float(conf.loc["const", 1]))
        slope_ci = (float(conf.loc[x_col, 0]), float(conf.loc[x_col, 1]))
    else:
        # ndarray인 경우: 행 0=const, 행 1=x_col, 열 0/1 = lo/hi
        intercept_ci = (float(conf[0, 0]), float(conf[0, 1]))
        slope_ci = (float(conf[1, 0]), float(conf[1, 1]))

    r2 = float(model.rsquared)

    return {
        "slope": slope,
        "slope_ci": slope_ci,
        "intercept": intercept,
        "intercept_ci": intercept_ci,
        "r2": r2,
        "summary": model.summary().as_text(),
    }



# --------------------------
# Main pipeline
# --------------------------
def run_pipeline(
    csv_path: str,
    bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 2025,
) -> None:
    logger.info("Loading data from: %s", csv_path)
    df = pd.read_csv(csv_path)
    df = _validate_and_standardize_columns(df)

    expert_cols = ["expert1", "expert2", "expert3"]
    logger.info("Computing panel consensus scores (mean / median / z-mean).")
    df = build_panel_scores(df, expert_cols)

    # --------------------------
    # Panel reliability (공통)
    # --------------------------
    logger.info("Estimating panel reliability: ICC(2,k) and Cronbach's alpha.")
    rel = panel_reliability(df, expert_cols)
    icc_tbl: pd.DataFrame = rel["icc_table"]

    logger.info("\n=== Dataset Summary (numeric columns) ===\n%s", numeric_summary(df))

    logger.info(
        "\n=== Panel Reliability ===\nICC(2,k): %.3f  95%% CI %s\nCronbach's α: %.3f",
        rel["icc2k"], rel["icc2k_ci"], rel["alpha"]
    )
    # Show condensed ICC table
    icc_view = icc_tbl.loc[
        icc_tbl["Type"].isin(["ICC1", "ICC2", "ICC3", "ICC1k", "ICC2k", "ICC3k"]),
        ["Type", "ICC", "CI95%", "F", "df1", "df2", "pval"],
    ]
    logger.info("\nICC table (selected rows):\n%s", icc_view.to_string(index=False))

    # --------------------------
    # 여러 RCF 컬럼 분석
    # --------------------------
    rcf_cols = [c for c in df.columns if c.startswith("rcf")]
    if not rcf_cols:
        raise ValueError("No RCF columns found in input data.")

    for rcf_col in rcf_cols:
        logger.info("\n\n#############################")
        logger.info("=== Results for %s ===", rcf_col)
        logger.info("#############################")

        # Correlation suite: RCF vs panel aggregates
        targets = ["panel_mean", "panel_median", "panel_zmean"]
        logger.info(
            "Computing rank correlations (Spearman, Kendall τ-b) with bootstrap 95%% CIs (B=%d).",
            bootstrap,
        )
        corr_tbl = correlation_suite(df, targets, x_col=rcf_col,
                                     B=bootstrap, alpha=alpha, seed=seed)

        # Per-rater correlations + Fisher z-mean
        logger.info("Computing per-rater correlations and Fisher's z-mean.")
        per_rater_tbl, rho_meta = per_rater_correlations(
            df, expert_cols, x_col=rcf_col, B=bootstrap, alpha=alpha, seed=seed
        )

        # Calibration regression (panel_mean ~ this RCF)
        logger.info("Fitting calibration regression: panel_mean ~ %s.", rcf_col)
        calib = calibration_regression(df, y_col="panel_mean", x_col=rcf_col)

        # --------------------------
        # Pretty logging outputs
        # --------------------------
        logger.info("\n=== %s vs Panel Consensus: Rank Correlations ===\n%s",
                    rcf_col, corr_tbl.to_string(index=False))

        logger.info("\n=== %s vs Individual Raters (Spearman) ===\n%s",
                    rcf_col, per_rater_tbl.to_string(index=False))
        logger.info("%s Fisher's z-mean (Spearman across raters): %.3f",
                    rcf_col, rho_meta)

        logger.info(
            "\n=== Calibration: OLS (panel_mean ~ %s) ===\n"
            "Slope: %.3f  95%% CI [%0.3f, %0.3f]\n"
            "Intercept: %.3f  95%% CI [%0.3f, %0.3f]\n"
            "R²: %.3f\n\nModel summary:\n%s",
            rcf_col,
            calib["slope"], calib["slope_ci"][0], calib["slope_ci"][1],
            calib["intercept"], calib["intercept_ci"][0], calib["intercept_ci"][1],
            calib["r2"], calib["summary"]
        )

        # Concise takeaway for papers
        try:
            best_idx = corr_tbl["Target"].tolist().index("panel_mean")
            best_row = corr_tbl.iloc[best_idx]
            logger.info(
                "\n=== Paper-Ready Takeaway for %s ===\n"
                "Panel reliability high (ICC(2,k)=%.2f, α=%.2f). "
                "%s shows moderate rank agreement with panel consensus "
                "(Spearman ρ=%s, 95%% CI %s; Kendall τ-b=%s, 95%% CI %s). "
                "Calibration R²=%.2f.",
                rcf_col,
                rel["icc2k"], rel["alpha"],
                rcf_col,
                best_row["Spearman ρ"], best_row["ρ 95% CI"],
                best_row["Kendall τ-b"], best_row["τ 95% CI"],
                calib["r2"]
            )
        except ValueError:
            logger.info("Could not locate 'panel_mean' row in correlation table for %s.", rcf_col)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the reliability of LLM-derived RCF against expert panel."
    )

    # project_root = {project_root} (evaluation 상위 폴더)
    project_root = Path(__file__).resolve().parent.parent
    default_csv = project_root / "data" / "recipe" / "evaluation" / "rcf_input.csv"

    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Path to the input CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap replicates for CI (default: 10000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for bootstrap (default: 2025)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(csv_path=str(args.csv), bootstrap=args.bootstrap, alpha=args.alpha, seed=args.seed)
