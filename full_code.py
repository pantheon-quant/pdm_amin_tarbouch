# %% CODE 1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECB Inflation Panel — HICP (ECB) with ADF, Granger, and VAR
===========================================================

Overview
--------
Single-file, reproducible script that builds a monthly inflation panel from the
ECB Data Portal (SDMX 2.1 REST) — HICP inflation (y/y, %) for multiple countries.

It then runs a compact time-series workflow suitable for teaching and quick diagnostics:
- ADF unit-root tests on inflation levels
- Bivariate Granger causality screening (predictors → target)
- Small VAR in levels with lag order selected by BIC

Key features
------------
- Uses official ECB SDMX API (no scraping)
- Explicit SDMX keys and dimensions documented in code
- Month indexing standardized to month-start timestamps for safe merges

Data source
-----------
ECB:  ECB Data Portal, dataset "ICP" (HICP).
      SDMX 2.1 REST pattern:
      https://data-api.ecb.europa.eu/service/data/ICP/{key}?format=csvdata&startPeriod=...&endPeriod=...

Econometric workflow
--------------------
- ADF test (H0: unit root) on each inflation series (levels)
- Granger causality tests (bivariate): does X help predict the target series?
  Ranking uses the minimum p-value across lags 1..maxlag
- VAR: target + top 2 Granger predictors; lag order chosen by BIC; VAR in levels

Outputs
-------
- Multi-line plot of the panel (incl. 0-line)
- Console tables:
  * ADF stats/p-values
  * Granger ranking
  * VAR lag selection (BIC) and estimation summary

Dependencies
------------
requests, pandas, numpy, matplotlib, statsmodels

Author / License
----------------
Eric Vansteenberghe (Banque de France)
Adapted: 2026-03-16 by Amin Tarbouch
License: MIT (recommended for teaching code)
"""

import requests
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR


# ============================================================
# 1. Fetch ECB HICP inflation panel
# ============================================================

def fetch_ecb_hicp_inflation_panel(
    countries,
    start="1997-01-01",
    end=None,
    item="000000",   # headline all-items HICP
    sa="N",          # neither seasonally nor working-day adjusted
    measure="4",     # percentage change (as used in ICP keys)
    variation="ANR", # annual rate of change
    freq="M",
    timeout=60
):
    """
    Fetch a monthly cross-country panel of HICP inflation (annual rate of change)
    from the ECB Data Portal (ICP dataflow).

    Returns
    -------
    panel_wide : pd.DataFrame
        Index: pandas datetime (monthly)
        Columns: country codes (e.g., DE, FR, IT)
        Values: inflation rate (float)
    raw_long : pd.DataFrame
        Long format with series dimensions, TIME_PERIOD and OBS_VALUE.
    """
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected response format. Columns: {list(raw.columns)}")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for cand in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if cand in raw.columns:
                country_col = cand
                break
    if country_col is None:
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [c for c in raw.columns if c not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the country column from the response.")
        country_col = nonstandard[0]

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )

    return panel, raw


# ============================================================
# 2. Example usage — EU-11 panel
# ============================================================

countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
infl_panel, infl_long = fetch_ecb_hicp_inflation_panel(
    countries=countries,
    start="2000-01",
    end="2025-12"
)

# Ensure month-start index
infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")


# ============================================================
# 3. Plot the inflation panel
# ============================================================

plt.figure(figsize=(12, 6))
for country in infl_panel.columns:
    plt.plot(infl_panel.index, infl_panel[country], label=country, linewidth=1)

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Time")
plt.ylabel("Inflation rate (y/y, %)")
plt.title("HICP Inflation Panel (ECB Data Portal)")
plt.legend(ncol=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


# ============================================================
# 4. Prepare data
# ============================================================

df = infl_panel.copy().sort_index().dropna()


# ============================================================
# 5. ADF unit-root tests
# ============================================================

print("\n=== ADF unit-root tests (levels) ===")
adf_results = []
for c in df.columns:
    stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
    adf_results.append({"country": c, "ADF_stat": stat, "pvalue": pval})

adf_table = pd.DataFrame(adf_results).sort_values("pvalue")
print(adf_table.to_string(index=False))


# ============================================================
# 6. Granger causality: X → FR
# ============================================================

target = "FR"
maxlag = 6

print(f"\n=== Granger causality tests: X → {target} ===")
granger_out = []

for c in df.columns:
    if c == target:
        continue

    data_gc = df[[target, c]]

    try:
        res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
        min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
        granger_out.append({"country": c, "min_pvalue": min_p})
    except Exception as e:
        print(f"Granger test failed for {c}: {e}")

granger_rank = (
    pd.DataFrame(granger_out)
    .sort_values("min_pvalue")
    .reset_index(drop=True)
)

print(f"\n=== Ranking of countries by Granger causality for {target} ===")
print(granger_rank.to_string(index=False))


# ============================================================
# 7. Simple VAR with BIC (FR + top 2 predictors)
# ============================================================

top_countries = granger_rank["country"].iloc[:2].tolist()
var_vars = [target] + top_countries

print("\nVAR variables:", var_vars)

X_var = df[var_vars]
model = VAR(X_var)
lag_selection = model.select_order(maxlags=6)
p = lag_selection.selected_orders["bic"]
p = max(1, p)

print("\n=== VAR lag selection (BIC) ===")
print(lag_selection.summary())
print(f"Selected lag order p = {p}")

var_res = model.fit(p)
print("\n=== VAR estimation results ===")
print(var_res.summary())

# %% CODE 2

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part A - PDM on European inflation dynamics
EU-11 version only (Ukraine excluded because of SSL fetch failure)

This script:
1. Loads infl_panel from ecb_hicp_panel_var_granger.py
2. Keeps only the 11 EU countries:
   DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, GR
3. Restricts to complete-case sample with .dropna()
4. Computes PDM pioneer weights using compute_pioneer_weights_angles
5. Plots:
   - one line per country over time
   - one heatmap
6. Builds average pioneer weights by subperiod
7. Ranks countries in each subperiod
8. Saves all outputs to an "outputs_partA_no_ukraine" folder
"""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from pdm import compute_pioneer_weights_angles


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("outputs_partA_no_ukraine")
OUTPUT_DIR.mkdir(exist_ok=True)

EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]

COUNTRY_NAMES = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "BE": "Belgium",
    "AT": "Austria",
    "PT": "Portugal",
    "IE": "Ireland",
    "FI": "Finland",
    "GR": "Greece",
}

PERIODS = {
    "I (2002-2007)": ("2002-01", "2007-12"),
    "II (2008-2012)": ("2008-01", "2012-12"),
    "III (2013-2019)": ("2013-01", "2019-12"),
    "IV (2020-2021)": ("2020-01", "2021-12"),
    "V (2022-2023)": ("2022-01", "2023-12"),
    "VI (2024-2025)": ("2024-01", "2025-12"),
}


# ============================================================
# Loading data
# ============================================================

def load_inflation_panel() -> pd.DataFrame:
    """
    Imports ecb_hicp_panel_var_granger.py and retrieves infl_panel.

    Important:
    This assumes that importing the module succeeds on your machine.
    If your original file still tries to fetch Ukraine on import, and crashes before
    infl_panel is created, then you will need to comment out the Ukraine-fetching part
    inside ecb_hicp_panel_var_granger.py itself.

    Once infl_panel exists, this script keeps only the EU-11 columns.
    """
    module = importlib.import_module("ecb_hicp_panel_var_granger")

    if not hasattr(module, "infl_panel"):
        raise AttributeError(
            "No variable named 'infl_panel' was found in "
            "ecb_hicp_panel_var_granger.py."
        )

    infl_panel = getattr(module, "infl_panel")

    if not isinstance(infl_panel, pd.DataFrame):
        raise TypeError("'infl_panel' exists but is not a pandas DataFrame.")

    panel = infl_panel.copy()

    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)

    missing_cols = [c for c in EU_COUNTRIES if c not in panel.columns]
    if missing_cols:
        raise ValueError(
            f"These required EU country columns are missing from infl_panel: {missing_cols}"
        )

    panel = panel[EU_COUNTRIES].copy()
    return panel


# ============================================================
# Analysis helpers
# ============================================================

def average_weights_by_period(weights: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """
    Returns a DataFrame:
    rows = countries
    columns = subperiods
    values = mean pioneer weight in that period
    """
    out = {}
    for period_name, (start, end) in periods.items():
        sub = weights.loc[start:end]
        out[period_name] = sub.mean(axis=0)
    return pd.DataFrame(out)


def rank_weights_by_period(avg_table: pd.DataFrame) -> pd.DataFrame:
    """
    1 = highest average pioneer weight in that subperiod
    """
    return avg_table.rank(axis=0, ascending=False, method="dense").astype(int)


def build_nonzero_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """
    For each country:
    - number of months with non-zero pioneer weight
    - share of months with non-zero pioneer weight
    - first and last non-zero date
    - average weight when non-zero
    - average weight full sample
    """
    rows = []

    for col in weights.columns:
        s = weights[col]
        mask = s > 0
        nonzero_dates = s.index[mask]

        rows.append(
            {
                "country": col,
                "country_name": COUNTRY_NAMES.get(col, col),
                "months_nonzero": int(mask.sum()),
                "share_nonzero": float(mask.mean()),
                "first_nonzero": nonzero_dates.min().strftime("%Y-%m") if len(nonzero_dates) else None,
                "last_nonzero": nonzero_dates.max().strftime("%Y-%m") if len(nonzero_dates) else None,
                "avg_weight_when_nonzero": float(s[mask].mean()) if mask.any() else 0.0,
                "avg_weight_full_sample": float(s.mean()),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["months_nonzero", "avg_weight_full_sample"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return out


# ============================================================
# Plotting
# ============================================================

def plot_line_chart(weights: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(14, 7))

    for col in weights.columns:
        plt.plot(weights.index, weights[col], linewidth=1.4, label=col)

    plt.title("PDM pioneer weights over time (angles) - EU-11 only")
    plt.xlabel("Date")
    plt.ylabel("Pioneer weight")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(weights: pd.DataFrame, output_path: Path) -> None:
    data = weights.T.values

    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title("PDM pioneer weights heatmap (angles) - EU-11 only")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    ax.set_yticks(np.arange(len(weights.columns)))
    ax.set_yticklabels(weights.columns)

    n_ticks = min(12, len(weights.index))
    tick_positions = np.linspace(0, len(weights.index) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [weights.index[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pioneer weight")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Text discussion templates
# ============================================================

def print_a1d_comment(weights: pd.DataFrame) -> None:
    """
    A short template for A.1(d)
    """
    stable = weights.loc["2000-01":"2007-12"]
    max_by_month = stable.max(axis=1)

    print("\n" + "=" * 80)
    print("A.1(d) DISCUSSION")
    print("=" * 80)
    print(
        "During 2000-2007, inflation was relatively low and stable. "
        "In the PDM framework, strong pioneers are less likely in such periods if countries move "
        "more synchronously and there is less directional divergence followed by convergence."
    )
    print()
    print(
        f"Average monthly maximum pioneer weight in 2000-2007: {max_by_month.mean():.6f}"
    )
    print(
        f"Maximum observed monthly pioneer weight in 2000-2007: {max_by_month.max():.6f}"
    )
    print()
    print(
        "Interpretation: if these values are lower than in crisis periods, this suggests that "
        "clear pioneers were less common during the Great Moderation. That is consistent with "
        "PDM theory: when panel members evolve in a more stable and homogeneous way, fewer countries "
        "stand out as early movers toward which the rest subsequently converges."
    )


def print_a2c_comment(avg_table: pd.DataFrame) -> None:
    """
    A short template for A.2(c)
    """
    print("\n" + "=" * 80)
    print("A.2(c) ECONOMIC INTERPRETATION")
    print("=" * 80)
    print(
        "Possible explanations for countries with high pioneer weights include differences in:"
    )
    print("- energy mix and exposure to imported energy shocks")
    print("- trade openness and speed of pass-through from foreign prices")
    print("- financial structure and credit transmission")
    print("- geographic position and logistics exposure")
    print("- sectoral composition, such as manufacturing, tourism, or shipping")
    print()
    print("Top 3 countries by subperiod:")
    for period in avg_table.columns:
        leaders = avg_table[period].sort_values(ascending=False).head(3)
        summary = ", ".join([f"{idx} ({val:.6f})" for idx, val in leaders.items()])
        print(f"{period}: {summary}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading inflation panel...")
    panel = load_inflation_panel()

    print("Restricting to complete-case sample with dropna()...")
    panel = panel.dropna().copy()

    print(f"Final sample shape: {panel.shape}")
    print(f"Start date: {panel.index.min().strftime('%Y-%m')}")
    print(f"End date:   {panel.index.max().strftime('%Y-%m')}")
    print(f"Countries:  {list(panel.columns)}")

    print("\nComputing angle-based pioneer weights...")
    w_angles = compute_pioneer_weights_angles(panel)

    if not isinstance(w_angles, pd.DataFrame):
        raise TypeError("compute_pioneer_weights_angles did not return a DataFrame.")

    if not isinstance(w_angles.index, pd.DatetimeIndex):
        w_angles.index = panel.index

    if list(w_angles.columns) != list(panel.columns):
        w_angles.columns = panel.columns

    # Save raw weights
    weights_csv = OUTPUT_DIR / "partA_pdm_angles_weights_eu11.csv"
    w_angles.to_csv(weights_csv)

    # Save plots
    print("Saving plots...")
    plot_line_chart(w_angles, OUTPUT_DIR / "partA_pioneer_weights_lines_eu11.png")
    plot_heatmap(w_angles, OUTPUT_DIR / "partA_pioneer_weights_heatmap_eu11.png")

    # Non-zero summary
    print("Building non-zero weight summary...")
    nonzero_summary = build_nonzero_summary(w_angles)
    nonzero_summary.to_csv(
        OUTPUT_DIR / "partA_nonzero_weight_summary_eu11.csv",
        index=False
    )

    print("\nA.1(c) Countries receiving non-zero pioneer weight:")
    print(
        nonzero_summary[
            ["country", "months_nonzero", "share_nonzero", "first_nonzero", "last_nonzero"]
        ].to_string(index=False)
    )

    # Averages by subperiod
    print("\nComputing average weights by subperiod...")
    avg_table = average_weights_by_period(w_angles, PERIODS)
    avg_table.to_csv(OUTPUT_DIR / "partA_average_weights_by_subperiod_eu11.csv")

    print("\nA.2(a) Average pioneer weights by subperiod:")
    print(avg_table.round(6).to_string())

    # Rankings
    ranking_table = rank_weights_by_period(avg_table)
    ranking_table.to_csv(OUTPUT_DIR / "partA_rankings_by_subperiod_eu11.csv")

    print("\nA.2(b) Rankings by subperiod (1 = highest average pioneer weight):")
    print(ranking_table.to_string())

    # Save top rankings as text
    with open(OUTPUT_DIR / "partA_top_rankings_eu11.txt", "w", encoding="utf-8") as f:
        for period in avg_table.columns:
            f.write(f"{period}\n")
            f.write(avg_table[period].sort_values(ascending=False).to_string())
            f.write("\n\n")

    # Print discussion templates
    print_a1d_comment(w_angles)
    print_a2c_comment(avg_table)

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

# %% CODE 3


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part A - PDM on European inflation dynamics
EU-11 version only (Ukraine excluded because of SSL fetch failure)

This script:
1. Loads infl_panel from ecb_hicp_panel_var_granger.py
2. Keeps only the 11 EU countries:
   DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, GR
3. Restricts to complete-case sample with .dropna()
4. Computes PDM pioneer weights using compute_pioneer_weights_angles
5. Plots:
   - one line per country over time
   - one heatmap
6. Builds average pioneer weights by subperiod
7. Ranks countries in each subperiod
8. Saves all outputs to an "outputs_partA_no_ukraine" folder
"""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import compute_pioneer_weights_angles


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("outputs_partA_no_ukraine")
OUTPUT_DIR.mkdir(exist_ok=True)

EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]

COUNTRY_NAMES = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "BE": "Belgium",
    "AT": "Austria",
    "PT": "Portugal",
    "IE": "Ireland",
    "FI": "Finland",
    "GR": "Greece",
}

PERIODS = {
    "I (2002-2007)": ("2002-01", "2007-12"),
    "II (2008-2012)": ("2008-01", "2012-12"),
    "III (2013-2019)": ("2013-01", "2019-12"),
    "IV (2020-2021)": ("2020-01", "2021-12"),
    "V (2022-2023)": ("2022-01", "2023-12"),
    "VI (2024-2025)": ("2024-01", "2025-12"),
}


# ============================================================
# Loading data
# ============================================================

def load_inflation_panel() -> pd.DataFrame:
    """
    Imports ecb_hicp_panel_var_granger.py and retrieves infl_panel.

    Important:
    This assumes that importing the module succeeds on your machine.
    If your original file still tries to fetch Ukraine on import, and crashes before
    infl_panel is created, then you will need to comment out the Ukraine-fetching part
    inside ecb_hicp_panel_var_granger.py itself.

    Once infl_panel exists, this script keeps only the EU-11 columns.
    """
    module = importlib.import_module("ecb_hicp_panel_var_granger")

    if not hasattr(module, "infl_panel"):
        raise AttributeError(
            "No variable named 'infl_panel' was found in "
            "ecb_hicp_panel_var_granger.py."
        )

    infl_panel = getattr(module, "infl_panel")

    if not isinstance(infl_panel, pd.DataFrame):
        raise TypeError("'infl_panel' exists but is not a pandas DataFrame.")

    panel = infl_panel.copy()

    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)

    missing_cols = [c for c in EU_COUNTRIES if c not in panel.columns]
    if missing_cols:
        raise ValueError(
            f"These required EU country columns are missing from infl_panel: {missing_cols}"
        )

    panel = panel[EU_COUNTRIES].copy()
    return panel


# ============================================================
# Analysis helpers
# ============================================================

def average_weights_by_period(weights: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """
    Returns a DataFrame:
    rows = countries
    columns = subperiods
    values = mean pioneer weight in that period
    """
    out = {}
    for period_name, (start, end) in periods.items():
        sub = weights.loc[start:end]
        out[period_name] = sub.mean(axis=0)
    return pd.DataFrame(out)


def rank_weights_by_period(avg_table: pd.DataFrame) -> pd.DataFrame:
    """
    1 = highest average pioneer weight in that subperiod
    """
    return avg_table.rank(axis=0, ascending=False, method="dense").astype(int)


def build_nonzero_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """
    For each country:
    - number of months with non-zero pioneer weight
    - share of months with non-zero pioneer weight
    - first and last non-zero date
    - average weight when non-zero
    - average weight full sample
    """
    rows = []

    for col in weights.columns:
        s = weights[col]
        mask = s > 0
        nonzero_dates = s.index[mask]

        rows.append(
            {
                "country": col,
                "country_name": COUNTRY_NAMES.get(col, col),
                "months_nonzero": int(mask.sum()),
                "share_nonzero": float(mask.mean()),
                "first_nonzero": nonzero_dates.min().strftime("%Y-%m") if len(nonzero_dates) else None,
                "last_nonzero": nonzero_dates.max().strftime("%Y-%m") if len(nonzero_dates) else None,
                "avg_weight_when_nonzero": float(s[mask].mean()) if mask.any() else 0.0,
                "avg_weight_full_sample": float(s.mean()),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["months_nonzero", "avg_weight_full_sample"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return out


# ============================================================
# Plotting
# ============================================================

def plot_line_chart(weights: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(14, 7))

    for col in weights.columns:
        plt.plot(weights.index, weights[col], linewidth=1.4, label=col)

    plt.title("PDM pioneer weights over time (angles) - EU-11 only")
    plt.xlabel("Date")
    plt.ylabel("Pioneer weight")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(weights: pd.DataFrame, output_path: Path) -> None:
    data = weights.T.values

    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title("PDM pioneer weights heatmap (angles) - EU-11 only")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    ax.set_yticks(np.arange(len(weights.columns)))
    ax.set_yticklabels(weights.columns)

    n_ticks = min(12, len(weights.index))
    tick_positions = np.linspace(0, len(weights.index) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [weights.index[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pioneer weight")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Text discussion templates
# ============================================================

def print_a1d_comment(weights: pd.DataFrame) -> None:
    """
    A short template for A.1(d)
    """
    stable = weights.loc["2000-01":"2007-12"]
    max_by_month = stable.max(axis=1)

    print("\n" + "=" * 80)
    print("A.1(d) DISCUSSION")
    print("=" * 80)
    print(
        "During 2000-2007, inflation was relatively low and stable. "
        "In the PDM framework, strong pioneers are less likely in such periods if countries move "
        "more synchronously and there is less directional divergence followed by convergence."
    )
    print()
    print(
        f"Average monthly maximum pioneer weight in 2000-2007: {max_by_month.mean():.6f}"
    )
    print(
        f"Maximum observed monthly pioneer weight in 2000-2007: {max_by_month.max():.6f}"
    )
    print()
    print(
        "Interpretation: if these values are lower than in crisis periods, this suggests that "
        "clear pioneers were less common during the Great Moderation. That is consistent with "
        "PDM theory: when panel members evolve in a more stable and homogeneous way, fewer countries "
        "stand out as early movers toward which the rest subsequently converges."
    )


def print_a2c_comment(avg_table: pd.DataFrame) -> None:
    """
    A short template for A.2(c)
    """
    print("\n" + "=" * 80)
    print("A.2(c) ECONOMIC INTERPRETATION")
    print("=" * 80)
    print(
        "Possible explanations for countries with high pioneer weights include differences in:"
    )
    print("- energy mix and exposure to imported energy shocks")
    print("- trade openness and speed of pass-through from foreign prices")
    print("- financial structure and credit transmission")
    print("- geographic position and logistics exposure")
    print("- sectoral composition, such as manufacturing, tourism, or shipping")
    print()
    print("Top 3 countries by subperiod:")
    for period in avg_table.columns:
        leaders = avg_table[period].sort_values(ascending=False).head(3)
        summary = ", ".join([f"{idx} ({val:.6f})" for idx, val in leaders.items()])
        print(f"{period}: {summary}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading inflation panel...")
    panel = load_inflation_panel()

    print("Restricting to complete-case sample with dropna()...")
    panel = panel.dropna().copy()

    print(f"Final sample shape: {panel.shape}")
    print(f"Start date: {panel.index.min().strftime('%Y-%m')}")
    print(f"End date:   {panel.index.max().strftime('%Y-%m')}")
    print(f"Countries:  {list(panel.columns)}")

    print("\nComputing angle-based pioneer weights...")
    w_angles = compute_pioneer_weights_angles(panel)

    if not isinstance(w_angles, pd.DataFrame):
        raise TypeError("compute_pioneer_weights_angles did not return a DataFrame.")

    if not isinstance(w_angles.index, pd.DatetimeIndex):
        w_angles.index = panel.index

    if list(w_angles.columns) != list(panel.columns):
        w_angles.columns = panel.columns

    # Save raw weights
    weights_csv = OUTPUT_DIR / "partA_pdm_angles_weights_eu11.csv"
    w_angles.to_csv(weights_csv)

    # Save plots
    print("Saving plots...")
    plot_line_chart(w_angles, OUTPUT_DIR / "partA_pioneer_weights_lines_eu11.png")
    plot_heatmap(w_angles, OUTPUT_DIR / "partA_pioneer_weights_heatmap_eu11.png")

    # Non-zero summary
    print("Building non-zero weight summary...")
    nonzero_summary = build_nonzero_summary(w_angles)
    nonzero_summary.to_csv(
        OUTPUT_DIR / "partA_nonzero_weight_summary_eu11.csv",
        index=False
    )

    print("\nA.1(c) Countries receiving non-zero pioneer weight:")
    print(
        nonzero_summary[
            ["country", "months_nonzero", "share_nonzero", "first_nonzero", "last_nonzero"]
        ].to_string(index=False)
    )

    # Averages by subperiod
    print("\nComputing average weights by subperiod...")
    avg_table = average_weights_by_period(w_angles, PERIODS)
    avg_table.to_csv(OUTPUT_DIR / "partA_average_weights_by_subperiod_eu11.csv")

    print("\nA.2(a) Average pioneer weights by subperiod:")
    print(avg_table.round(6).to_string())

    # Rankings
    ranking_table = rank_weights_by_period(avg_table)
    ranking_table.to_csv(OUTPUT_DIR / "partA_rankings_by_subperiod_eu11.csv")

    print("\nA.2(b) Rankings by subperiod (1 = highest average pioneer weight):")
    print(ranking_table.to_string())

    # Save top rankings as text
    with open(OUTPUT_DIR / "partA_top_rankings_eu11.txt", "w", encoding="utf-8") as f:
        for period in avg_table.columns:
            f.write(f"{period}\n")
            f.write(avg_table[period].sort_values(ascending=False).to_string())
            f.write("\n\n")

    # Print discussion templates
    print_a1d_comment(w_angles)
    print_a2c_comment(avg_table)

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
# %%
