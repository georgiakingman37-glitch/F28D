import io
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# BASIC CONFIG
# =========================
st.set_page_config(page_title="DoorDash Alcohol – Early-Life Flags", layout="wide")

# ---- Column mapping (edit these keys to match Sigma export headers) ----
# Right side should exactly match your Sigma column names.
COLUMN_MAP: Dict[str, str] = {
    "merchant_id": "merchant_id",
    "merchant_name": "merchant_name",
    "go_live_ts": "go_live_timestamp",           # e.g., 2025-09-14 10:15:00
    "order_id": "order_id",
    "order_created_ts": "order_created_timestamp",
    "is_cancelled": "is_cancelled",              # boolean or 0/1
    "cancel_initiator": "cancel_initiator",      # 'merchant', 'customer', 'dasher', etc.
    # OPTIONAL columns:
    "uptime_pct": "uptime_pct",                  # (optional) 0–1 or 0–100; averaged over first 28 days
    "owner_name": "account_owner",               # (optional) store/account owner name(s)
}

MERCHANT_CANCEL_VALUES = {"merchant", "mx", "store", "merchant_initiated"}

# =========================
# HELPERS
# =========================
def _normalize_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "t", "1", "y", "yes"}
    false_vals = {"false", "f", "0", "n", "no"}
    out = []
    for v in s:
        if v in true_vals:
            out.append(True)
        elif v in false_vals:
            out.append(False)
        else:
            try:
                out.append(float(v) > 0)
            except Exception:
                out.append(False)
    return pd.Series(out, index=series.index, dtype=bool)

def _coerce_ts(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        return dt.dt.tz_convert(None)
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def _weeks_since(go_live: pd.Timestamp, ts: pd.Series) -> pd.Series:
    days = (ts - go_live).dt.total_seconds() / 86400
    return np.floor(days / 7).astype("Int64") + 1

def _merchant_mx_rate(df_28d: pd.DataFrame) -> Tuple[int, int, float]:
    if df_28d.empty:
        return 0, 0, 0.0
    cancelled = df_28d["is_cancelled"].fillna(False)
    initiator = df_28d["cancel_initiator"].astype(str).str.lower().str.strip()
    mx = cancelled & initiator.isin(MERCHANT_CANCEL_VALUES)
    total = len(df_28d)
    mx_count = int(mx.sum())
    return total, mx_count, (mx_count / total) if total else 0.0

def _avg_uptime_28d(df_28d: pd.DataFrame) -> Optional[float]:
    if "uptime_pct" not in df_28d.columns or df_28d.empty:
        return None
    vals = pd.to_numeric(df_28d["uptime_pct"], errors="coerce")
    if vals.dropna().max() > 1.0:
        vals = vals / 100.0
    if vals.dropna().empty:
        return None
    return float(vals.dropna().mean())

# ---- NEW: owner filtering helpers ----
def _owner_match_mask(series: pd.Series, query: str) -> pd.Series:
    """
    Match owner names case-insensitively, splitting multi-owner cells
    on common delimiters like ; , & / |
    """
    if series is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    q = (query or "").strip().lower()
    if not q:
        return pd.Series(True, index=series.index)  # no filter
    # normalize owners into tokens
    sep_pattern = r"[;,&/|]+"
    # lower + strip
    s = series.fillna("").astype(str).str.lower()
    # quick contains OR any token match
    contains = s.str.contains(q, na=False)
    # tokenized exact-ish component match
    tokens = s.str.split(sep_pattern)
    token_match = tokens.apply(lambda parts: any(q in p.strip() for p in parts if p))
    return contains | token_match

# =========================
# SESSION STORE
# =========================
if "orders" not in st.session_state:
    st.session_state["orders"] = pd.DataFrame(columns=list(COLUMN_MAP.values()))

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Upload weekly Sigma export")
    uploaded = st.file_uploader(
        "CSV or Parquet",
        type=["csv", "parquet"],
        accept_multiple_files=False
    )
    dedupe_cols = st.multiselect(
        "De-duplicate on",
        options=[
            COLUMN_MAP["order_id"],
            COLUMN_MAP["merchant_id"],
            COLUMN_MAP["order_created_ts"]
        ],
        default=[COLUMN_MAP["order_id"]]
    )
    ing_btn = st.button("Ingest / Append Data")

    st.markdown("---")
    st.caption("Thresholds")
    osw_min = st.number_input("Orders per week minimum (OSW)", min_value=1, value=5)
    mx_threshold = st.number_input("Egregious Mx threshold (%)", min_value=0.0, value=6.0, step=0.5) / 100.0
    uptime_min = st.number_input("Low uptime threshold (%)", min_value=0.0, value=90.0, step=1.0) / 100.0
    days_7 = 7
    days_28 = 28

    st.markdown("---")
    st.caption("Owner self-serve")
    owner_query = st.text_input("Filter by Account Owner name", placeholder="e.g., Jane Doe")

    st.markdown("---")
    st.caption("Schema mapping")
    st.json(COLUMN_MAP, expanded=False)

# =========================
# INGESTION
# =========================
if uploaded and ing_btn:
    if uploaded.name.lower().endswith(".csv"):
        new_df = pd.read_csv(uploaded)
    else:
        new_df = pd.read_parquet(uploaded)

    required = ["merchant_id","merchant_name","go_live_ts","order_id","order_created_ts","is_cancelled","cancel_initiator"]
    missing = [COLUMN_MAP[k] for k in required if COLUMN_MAP[k] not in new_df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        df = new_df.copy()

        # Normalize types
        df[COLUMN_MAP["go_live_ts"]] = _coerce_ts(df[COLUMN_MAP["go_live_ts"]])
        df[COLUMN_MAP["order_created_ts"]] = _coerce_ts(df[COLUMN_MAP["order_created_ts"]])
        df[COLUMN_MAP["is_cancelled"]] = _normalize_bool(df[COLUMN_MAP["is_cancelled"]])

        if COLUMN_MAP["uptime_pct"] in df.columns:
            df[COLUMN_MAP["uptime_pct"]] = pd.to_numeric(df[COLUMN_MAP["uptime_pct"]], errors="coerce")

        # Keep valid rows
        df = df[
            df[COLUMN_MAP["go_live_ts"]].notna() &
            df[COLUMN_MAP["order_created_ts"]].notna()
        ].copy()

        combined = pd.concat([st.session_state["orders"], df], ignore_index=True)
        if dedupe_cols:
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
        st.session_state["orders"] = combined
        st.success(f"Ingested {len(df):,} rows. Current store: {len(combined):,} rows.")

# =========================
# METRICS ENGINE
# =========================
def compute_early_metrics(df_orders: pd.DataFrame, merchant_id: Optional[str] = None, merchant_name: Optional[str] = None) -> Dict:
    col = COLUMN_MAP
    if merchant_id:
        msk = df_orders[col["merchant_id"]].astype(str) == str(merchant_id)
    elif merchant_name:
        msk = df_orders[col["merchant_name"]].astype(str).str.lower().str.contains(merchant_name.lower(), na=False)
    else:
        raise ValueError("Pass merchant_id or merchant_name")

    m = df_orders.loc[msk].copy()
    if m.empty:
        return {"found": False}

    merchant_name_guess = (
        m[col["merchant_name"]].dropna().astype(str).value_counts().idxmax()
        if m[col["merchant_name"]].notna().any() else None
    )

    go_live = m[col["go_live_ts"]].min()
    end_7d = go_live + timedelta(days=days_7)
    end_28d = go_live + timedelta(days=days_28)

    w7 = m[(m[col["order_created_ts"]] >= go_live) & (m[col["order_created_ts"]] < end_7d)].copy()
    w28 = m[(m[col["order_created_ts"]] >= go_live) & (m[col["order_created_ts"]] < end_28d)].copy()

    has_7d_order = len(w7) > 0
    has_28d_order = len(w28) > 0

    m["week_index"] = _weeks_since(go_live, m[col["order_created_ts"]])
    first4 = m[(m["week_index"] >= 1) & (m["week_index"] <= 4)].copy()
    weekly_counts = first4.groupby("week_index")[col["order_id"]].nunique().reindex([1,2,3,4], fill_value=0)
    osw_pass = bool((weekly_counts >= osw_min).all())

    total, mx_count, mx_rate = _merchant_mx_rate(
        w28.assign(
            is_cancelled=w28[col["is_cancelled"]],
            cancel_initiator=w28[col["cancel_initiator"]]
        )
    )
    mx_flag = mx_rate > mx_threshold

    if COLUMN_MAP["uptime_pct"] in m.columns:
        avg_uptime = _avg_uptime_28d(w28.rename(columns={col["uptime_pct"]: "uptime_pct"}))
    else:
        avg_uptime = None
    low_uptime = (avg_uptime is not None) and (avg_uptime < uptime_min)

    flags = {
        "no_order_7d": not has_7d_order,
        "no_order_28d": not has_28d_order,
        "low_uptime": low_uptime,
        "mx_egregious": mx_flag,
        "osw_fail": not osw_pass,
    }

    return {
        "found": True,
        "merchant_id": str(m[col["merchant_id"]].iloc[0]),
        "merchant_name": merchant_name_guess,
        "go_live": go_live,
        "window_7d_end": end_7d,
        "window_28d_end": end_28d,
        "weekly_counts": weekly_counts.to_dict(),
        "has_7d_order": has_7d_order,
        "has_28d_order": has_28d_order,
        "osw_pass": osw_pass,
        "mx_total": total,
        "mx_count": mx_count,
        "mx_rate": mx_rate,
        "avg_uptime": avg_uptime,
        "flags": flags
    }

def summarize_portfolio(df_orders: pd.DataFrame) -> pd.DataFrame:
    mid_col = COLUMN_MAP["merchant_id"]
    gl = (df_orders.groupby(mid_col, as_index=False)[COLUMN_MAP["go_live_ts"]].min()
          .rename(columns={COLUMN_MAP["go_live_ts"]: "go_live"}))

    rows = []
    for _, row in gl.iterrows():
        mid = str(row[mid_col])
        m = compute_early_metrics(df_orders, merchant_id=mid)
        if m["found"]:
            rows.append({
                "merchant_id": m["merchant_id"],
                "merchant_name": m["merchant_name"],
                "go_live": m["go_live"],
                "has_order_7d": m["has_7d_order"],
                "has_order_28d": m["has_28d_order"],
                "w1_orders": m["weekly_counts"].get(1, 0),
                "w2_orders": m["weekly_counts"].get(2, 0),
                "w3_orders": m["weekly_counts"].get(3, 0),
                "w4_orders": m["weekly_counts"].get(4, 0),
                "osw_pass": m["osw_pass"],
                "mx_orders_in_window": m["mx_total"],
                "mx_merchant_cancels": m["mx_count"],
                "mx_rate": m["mx_rate"],
                "avg_uptime": m["avg_uptime"],
                "flag_no_order_7d": m["flags"]["no_order_7d"],
                "flag_no_order_28d": m["flags"]["no_order_28d"],
                "flag_low_uptime": m["flags"]["low_uptime"],
                "flag_mx_egregious": m["flags"]["mx_egregious"],
                "flag_osw_fail": m["flags"]["osw_fail"],
            })
    return pd.DataFrame(rows)

# =========================
# MAIN
# =========================
st.title("🍷 Alcohol – Early-Life Flags (First 28 Days)")
data = st.session_state["orders"]

if data.empty:
    st.info("Upload your Sigma export to begin.")
    st.stop()

# Normalize working frame to canonical names used internally
df = data.rename(columns={
    COLUMN_MAP["merchant_id"]: "merchant_id",
    COLUMN_MAP["merchant_name"]: "merchant_name",
    COLUMN_MAP["go_live_ts"]: "go_live_ts",
    COLUMN_MAP["order_id"]: "order_id",
    COLUMN_MAP["order_created_ts"]: "order_created_ts",
    COLUMN_MAP["is_cancelled"]: "is_cancelled",
    COLUMN_MAP["cancel_initiator"]: "cancel_initiator",
    COLUMN_MAP.get("uptime_pct","uptime_pct"): "uptime_pct",
    COLUMN_MAP.get("owner_name","owner_name"): "owner_name",
}).copy()

# ---- NEW: apply owner filter across the app ----
if "owner_name" in df.columns and df["owner_name"].notna().any() and (owner_query or "").strip():
    mask = _owner_match_mask(df["owner_name"], owner_query)
    df = df.loc[mask].copy()
    st.info(f"Filtering by **Owner contains:** “{owner_query.strip()}”. Showing only your accounts.")
elif (owner_query or "").strip():
    st.warning("Owner filter provided, but no owner column found in data. Set COLUMN_MAP['owner_name'] to your Sigma header to enable owner filtering.")

summary = summarize_portfolio(df)
if summary.empty:
    st.warning("No merchants summarized (after filters). Check column mapping, timestamps, and owner filter.")
    st.stop()

# =========================
# PORTFOLIO OVERVIEW (SIMPLE)
# =========================
st.subheader("Portfolio Overview (Simple)")

total_merchants = len(summary)
pct = lambda n: f"{(n / total_merchants * 100):.0f}%" if total_merchants else "0%"

no_7d = int((~summary["has_order_7d"]).sum())
no_28d = int((~summary["has_order_28d"]).sum())
mx_bad = int((summary["mx_rate"] > mx_threshold).sum())
osw_bad = int((~summary["osw_pass"]).sum())
low_up = int(summary["avg_uptime"].lt(uptime_min).fillna(False).sum()) if "avg_uptime" in summary else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("No Order in 7 Days", f"{no_7d:,}", pct(no_7d))
c2.metric("No Order in 28 Days", f"{no_28d:,}", pct(no_28d))
c3.metric("Egregious Mx (> threshold)", f"{mx_bad:,}", pct(mx_bad))
c4.metric("OSW Fail (< target)", f"{osw_bad:,}", pct(osw_bad))
c5.metric("Low Uptime", f"{low_up:,}", pct(low_up))

# Compact bar: count of merchants by # of failing flags
flag_cols = ["flag_no_order_7d","flag_no_order_28d","flag_low_uptime","flag_mx_egregious","flag_osw_fail"]
summary["fail_count"] = summary[flag_cols].sum(axis=1)
counts = summary["fail_count"].value_counts().sort_index()
st.bar_chart(counts.rename_axis("# Flags").rename("Merchants"))

# =========================
# FLAGGED ACCOUNTS (ONLY)
# =========================
st.markdown("---")
st.subheader("Flagged Accounts (Needs Attention)")

flagged = summary[summary[flag_cols].any(axis=1)].copy()

def _fmt_rate(x):
    return "" if pd.isna(x) else f"{x*100:.1f}%"

def _fmt_uptime(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x*100:.1f}%"

if flagged.empty:
    st.success("Great news — no flagged accounts under current thresholds.")
else:
    simple_cols = [
        "merchant_name","merchant_id","go_live",
        "has_order_7d","has_order_28d",
        "w1_orders","w2_orders","w3_orders","w4_orders",
        "osw_pass","mx_rate","avg_uptime",
        "flag_no_order_7d","flag_no_order_28d","flag_low_uptime","flag_mx_egregious","flag_osw_fail"
    ]
    show = flagged.loc[:, simple_cols].rename(columns={
        "has_order_7d":"7d order?",
        "has_order_28d":"28d order?",
        "osw_pass":"OSW pass",
        "mx_rate":"Mx rate",
        "avg_uptime":"Avg uptime (28d)"
    }).copy()

    show["Mx rate"] = show["Mx rate"].map(_fmt_rate)
    show["Avg uptime (28d)"] = show["Avg uptime (28d)"].map(_fmt_uptime)

    st.dataframe(show, use_container_width=True)

# =========================
# MERCHANT LOOKUP (SIMPLE + CLEAR)
# =========================
st.markdown("---")
st.subheader("Merchant Lookup")

lc1, lc2, lc3 = st.columns([1.2,1.2,0.8])
with lc1:
    q_id = st.text_input("Merchant ID (exact)")
with lc2:
    q_name = st.text_input("Merchant Name (contains)")
with lc3:
    go = st.button("Check")

if go:
    try:
        res = compute_early_metrics(df, merchant_id=q_id.strip() or None, merchant_name=q_name.strip() or None)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if not res["found"]:
        st.warning("No matching merchant.")
    else:
        left, right = st.columns([1,1])
        with left:
            st.markdown(f"**Merchant:** {res['merchant_name']}  \n**ID:** `{res['merchant_id']}`")
            st.markdown(f"**Go-Live:** {pd.to_datetime(res['go_live']).strftime('%Y-%m-%d')}")
            st.markdown(f"**Windows:** 7d → {res['window_7d_end'].date()} • 28d → {res['window_28d_end'].date()}")
        with right:
            st.metric("Mx rate (28d)", f"{res['mx_rate']*100:.2f}%")
            st.metric("Avg uptime (28d)", "N/A" if res["avg_uptime"] is None else f"{res['avg_uptime']*100:.1f}%")

        st.markdown("### KPI Checks")
        checks = pd.DataFrame({
            "KPI": [
                "Has order in 7 days",
                "Has order in 28 days",
                f"OSW: each of weeks 1–4 ≥ {osw_min}",
                f"Mx rate ≤ {mx_threshold*100:.1f}%",
                f"Avg uptime ≥ {uptime_min*100:.0f}%"
            ],
            "Status": [
                "✅" if res["has_7d_order"] else "❌",
                "✅" if res["has_28d_order"] else "❌",
                "✅" if res["osw_pass"] else "❌",
                "✅" if res["mx_rate"] <= mx_threshold else "❌",
                "✅" if (res["avg_uptime"] is not None and res["avg_uptime"] >= uptime_min) else ("N/A" if res["avg_uptime"] is None else "❌")
            ]
        })
        st.dataframe(checks, hide_index=True, use_container_width=True)

        st.markdown("### Weekly Order Counts (W1–W4)")
        wk = pd.Series(res["weekly_counts"]).reindex([1,2,3,4], fill_value=0)
        st.bar_chart(wk.rename({1:"W1",2:"W2",3:"W3",4:"W4"}))

# =========================
# DOWNLOADS
# =========================
st.markdown("---")
csv_all = summary.to_csv(index=False).encode("utf-8")
csv_flags = flagged.to_csv(index=False).encode("utf-8") if not flagged.empty else None

d1, d2 = st.columns(2)
d1.download_button(
    "Download Portfolio Summary (CSV)",
    data=csv_all,
    file_name=f"early_life_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)
d2.download_button(
    "Download Flagged Accounts (CSV)",
    data=csv_flags if csv_flags is not None else "".encode(),
    file_name=f"flagged_accounts_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    disabled=flagged.empty
)

st.caption("Tip: Set COLUMN_MAP['owner_name'] to your Sigma header (e.g., 'account_owner') to enable owner filtering. Users can type their name to see only their stores.")
