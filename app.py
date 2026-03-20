"""
Flask app for Weibull and DS-Weibull lifetime prediction analysis.
"""

import io
import json
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, jsonify, render_template, request

from weibull_analysis import (
    WeibullAnalysis,
    DSWeibullAnalysis,
    weibull_plot_data,
    parse_input_data,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        raw_text = data.get("text", "").strip()
        model_type = data.get("model", "weibull")   # "weibull" | "ds_weibull" | "both"
        cl = float(data.get("confidence_level", 0.9))
        t_predict = data.get("t_predict", "")       # comma-separated times for prediction
        b_lives = data.get("b_lives", "10,50,90")   # comma-separated percentiles

        if not raw_text:
            return jsonify({"error": "データが入力されていません。"}), 400

        times, censored = parse_input_data(raw_text)

        if len(times) == 0:
            return jsonify({"error": "有効なデータが見つかりません。"}), 400

        if np.any(times <= 0):
            return jsonify({"error": "すべての時間は正の値である必要があります。"}), 400

        n_fail = int(np.sum(~censored))
        n_cens = int(np.sum(censored))

        # Parse B-life percentiles
        try:
            b_list = [float(x.strip()) / 100.0 for x in b_lives.split(",") if x.strip()]
            b_list = [p for p in b_list if 0 < p < 1]
        except Exception:
            b_list = [0.1, 0.5, 0.9]

        # Parse prediction times
        t_pred_arr = []
        if t_predict.strip():
            try:
                t_pred_arr = [float(x.strip()) for x in t_predict.split(",") if x.strip()]
            except Exception:
                pass

        result = {
            "n_total": len(times),
            "n_failures": n_fail,
            "n_censored": n_cens,
        }

        # ── Fit models ──────────────────────────────────────────
        if model_type in ("weibull", "both"):
            wb = WeibullAnalysis()
            wb.fit(times, censored)
            result["weibull"] = _weibull_result(wb, times, censored, cl, b_list, t_pred_arr)

        if model_type in ("ds_weibull", "both"):
            ds = DSWeibullAnalysis()
            ds.fit(times, censored)
            result["ds_weibull"] = _ds_weibull_result(ds, times, censored, cl, b_list, t_pred_arr)

        # Model comparison (AIC/BIC) when both fitted
        if model_type == "both" and "weibull" in result and "ds_weibull" in result:
            result["comparison"] = {
                "weibull_aic": result["weibull"]["aic"],
                "weibull_bic": result["weibull"]["bic"],
                "ds_weibull_aic": result["ds_weibull"]["aic"],
                "ds_weibull_bic": result["ds_weibull"]["bic"],
                "better_aic": "DS-Weibull" if result["ds_weibull"]["aic"] < result["weibull"]["aic"] else "Weibull",
                "better_bic": "DS-Weibull" if result["ds_weibull"]["bic"] < result["weibull"]["bic"] else "Weibull",
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/point_predict", methods=["POST"])
def point_predict():
    """
    Lightweight endpoint: given already-fitted params, compute F(t) or t(F) instantly.
    Accepts: { model, params, query_t, query_F, cl }
    Returns: { F_at_t, ci_lo_F, ci_hi_F, t_at_F, ci_lo_t, ci_hi_t }
    """
    try:
        data = request.get_json(force=True)
        model = data.get("model", "weibull")
        params = data.get("params", {})
        cl = float(data.get("cl", 0.9))
        query_t = data.get("query_t")   # float or null
        query_F = data.get("query_F")   # float (0-100) or null

        result = {}

        if model == "weibull":
            wb = WeibullAnalysis()
            wb.alpha = float(params["alpha"])
            wb.beta  = float(params["beta"])
            wb.mu    = np.log(wb.alpha)
            wb.sigma = 1.0 / wb.beta
            cov = params.get("cov")
            wb.cov = np.array(cov) if cov is not None else None

            if query_t is not None:
                t = float(query_t)
                F = float(wb.cdf(t))
                lo_arr, hi_arr = wb.cdf_ci(np.array([t]), cl)
                result["F_at_t"] = F * 100
                result["ci_lo_F"] = float(lo_arr[0]) * 100
                result["ci_hi_F"] = float(hi_arr[0]) * 100

            if query_F is not None:
                p = float(query_F) / 100.0
                if 0 < p < 1:
                    t_p = float(wb.quantile(p))
                    lo_t, hi_t = wb.quantile_ci(p, cl)
                    result["t_at_F"] = t_p
                    result["ci_lo_t"] = float(lo_t)
                    result["ci_hi_t"] = float(hi_t)

        elif model == "ds_weibull":
            ds = DSWeibullAnalysis()
            ds.alpha = float(params["alpha"])
            ds.beta  = float(params["beta"])
            ds.DS    = float(params["DS"])
            ds.mu    = np.log(ds.alpha)
            ds.sigma = 1.0 / ds.beta
            cov = params.get("cov")
            ds.cov = np.array(cov) if cov is not None else None

            if query_t is not None:
                t = float(query_t)
                F = float(ds.cdf(t))
                lo_arr, hi_arr = ds.cdf_ci(np.array([t]), cl)
                result["F_at_t"] = F * 100
                result["ci_lo_F"] = float(lo_arr[0]) * 100
                result["ci_hi_F"] = float(hi_arr[0]) * 100

            if query_F is not None:
                p = float(query_F) / 100.0
                if 0 < p < ds.DS:
                    t_p = float(ds.quantile(p))
                    lo_t, hi_t = ds.quantile_ci(p, cl)
                    result["t_at_F"] = t_p
                    result["ci_lo_t"] = float(lo_t)
                    result["ci_hi_t"] = float(hi_t)
                elif p >= ds.DS:
                    result["t_at_F"] = None
                    result["t_at_F_note"] = f"F% ≥ DS({ds.DS*100:.1f}%) — 到達しない"

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Accept CSV file upload, return parsed data as text for the textarea."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "ファイルが見つかりません。"}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "ファイルが選択されていません。"}), 400

        content = f.read().decode("utf-8-sig")  # handle BOM
        df = pd.read_csv(io.StringIO(content))

        # Detect time column and optional censor column
        time_col = None
        cens_col = None

        lower = [c.lower() for c in df.columns]
        for cname in ["time", "t", "lifetime", "duration", "寿命", "時間", "age"]:
            for i, lc in enumerate(lower):
                if cname in lc:
                    time_col = df.columns[i]
                    break
            if time_col:
                break
        if time_col is None and len(df.columns) >= 1:
            time_col = df.columns[0]

        for cname in ["cens", "censored", "status", "failed", "failure", "suspend", "打ち切り", "censure"]:
            for i, lc in enumerate(lower):
                if cname in lc:
                    cens_col = df.columns[i]
                    break
            if cens_col:
                break
        if cens_col is None and len(df.columns) >= 2:
            # second column might be censor flag
            col2 = df.columns[1]
            unique_vals = df[col2].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0, "0", "1"}):
                cens_col = col2

        lines = []
        for _, row in df.iterrows():
            try:
                t = float(row[time_col])
                if cens_col is not None:
                    c = int(float(row[cens_col]))
                    lines.append(f"{t}\t{c}")
                else:
                    lines.append(f"{t}")
            except (ValueError, KeyError):
                continue

        text = "\n".join(lines)
        columns_info = f"時間列: {time_col}" + (f"、打ち切り列: {cens_col}" if cens_col else "（打ち切りなし）")
        return jsonify({"text": text, "info": columns_info, "rows": len(lines)})

    except Exception as e:
        return jsonify({"error": f"CSVパースエラー: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────
# Result builders
# ──────────────────────────────────────────────────────────────

def _weibull_result(wb, times, censored, cl, b_list, t_pred_arr):
    """Build JSON-serializable result dict for Weibull."""
    alpha_ci, beta_ci = wb.parameter_ci(cl)
    n_fail = int(np.sum(~censored))
    n = len(times)

    # B-lives
    b_table = []
    for p in b_list:
        t_p = float(wb.quantile(p))
        lo, hi = wb.quantile_ci(p, cl)
        b_table.append({
            "B": f"B{p*100:.0f}",
            "p": p,
            "time": _fmt(t_p),
            "ci_lo": _fmt(float(lo)),
            "ci_hi": _fmt(float(hi)),
        })

    # CDF at prediction times
    pred_table = []
    for t in t_pred_arr:
        F = float(wb.cdf(t))
        lo_arr, hi_arr = wb.cdf_ci(np.array([t]), cl)
        pred_table.append({
            "t": _fmt(t),
            "F": _fmt(F * 100) + " %",
            "ci_lo": _fmt(float(lo_arr[0]) * 100) + " %",
            "ci_hi": _fmt(float(hi_arr[0]) * 100) + " %",
        })

    # Plots
    plots = _make_weibull_plots(wb, times, censored, cl)

    # Raw params for client-side point_predict
    cov_list = wb.cov.tolist() if wb.cov is not None and not np.any(np.isnan(wb.cov)) else None

    return {
        "alpha": _fmt(wb.alpha),
        "beta": _fmt(wb.beta),
        "alpha_raw": float(wb.alpha),
        "beta_raw": float(wb.beta),
        "alpha_ci": [_fmt(alpha_ci[0]), _fmt(alpha_ci[1])],
        "beta_ci": [_fmt(beta_ci[0]), _fmt(beta_ci[1])],
        "mean": _fmt(wb.mean()),
        "log_likelihood": _fmt(wb.log_likelihood),
        "aic": _fmt(wb.aic),
        "bic": _fmt(wb.bic),
        "converged": wb.converged,
        "b_table": b_table,
        "pred_table": pred_table,
        "plots": plots,
        "cl_pct": int(cl * 100),
        "params_raw": {"alpha": float(wb.alpha), "beta": float(wb.beta), "cov": cov_list},
    }


def _ds_weibull_result(ds, times, censored, cl, b_list, t_pred_arr):
    """Build JSON-serializable result dict for DS-Weibull."""
    alpha_ci, beta_ci, DS_ci = ds.parameter_ci(cl)

    # B-lives (only valid for p < DS)
    b_table = []
    for p in b_list:
        if p >= ds.DS:
            b_table.append({
                "B": f"B{p*100:.0f}", "p": p,
                "time": "∞ (p ≥ DS)",
                "ci_lo": "—", "ci_hi": "—",
            })
        else:
            t_p = float(ds.quantile(p))
            lo, hi = ds.quantile_ci(p, cl)
            b_table.append({
                "B": f"B{p*100:.0f}", "p": p,
                "time": _fmt(t_p),
                "ci_lo": _fmt(float(lo)),
                "ci_hi": _fmt(float(hi)),
            })

    pred_table = []
    for t in t_pred_arr:
        F = float(ds.cdf(t))
        lo_arr, hi_arr = ds.cdf_ci(np.array([t]), cl)
        pred_table.append({
            "t": _fmt(t),
            "F": _fmt(F * 100) + " %",
            "ci_lo": _fmt(float(lo_arr[0]) * 100) + " %",
            "ci_hi": _fmt(float(hi_arr[0]) * 100) + " %",
        })

    plots = _make_ds_plots(ds, times, censored, cl)

    cov_list = ds.cov.tolist() if ds.cov is not None and not np.any(np.isnan(ds.cov)) else None

    return {
        "alpha": _fmt(ds.alpha),
        "beta": _fmt(ds.beta),
        "DS": _fmt(ds.DS * 100) + " %",
        "DS_raw": _fmt(ds.DS),
        "alpha_ci": [_fmt(alpha_ci[0]), _fmt(alpha_ci[1])],
        "beta_ci": [_fmt(beta_ci[0]), _fmt(beta_ci[1])],
        "DS_ci": [_fmt(DS_ci[0] * 100) + " %", _fmt(DS_ci[1] * 100) + " %"],
        "log_likelihood": _fmt(ds.log_likelihood),
        "aic": _fmt(ds.aic),
        "bic": _fmt(ds.bic),
        "converged": ds.converged,
        "b_table": b_table,
        "pred_table": pred_table,
        "plots": plots,
        "cl_pct": int(cl * 100),
        "params_raw": {"alpha": float(ds.alpha), "beta": float(ds.beta), "DS": float(ds.DS), "cov": cov_list},
    }


# ──────────────────────────────────────────────────────────────
# Plot builders
# ──────────────────────────────────────────────────────────────

def _make_weibull_plots(wb, times, censored, cl):
    """Return Plotly JSON strings for Weibull probability plot + CDF + PDF + HF."""
    t_fail, mr, x_emp, y_emp = weibull_plot_data(times, censored)
    t_cens = times[censored] if np.any(censored) else np.array([])
    n = len(times)

    t_min = max(times.min() * 0.1, 1e-6)
    t_max = times.max() * 2
    t_line = np.logspace(np.log10(t_min), np.log10(t_max), 300)

    # ── 1. Weibull Probability Plot ─────────────────────────────
    fig_prob = go.Figure()

    # Empirical points (failures)
    if len(t_fail) > 0:
        fig_prob.add_trace(go.Scatter(
            x=t_fail.tolist(), y=mr.tolist(),
            mode="markers",
            marker=dict(symbol="circle", size=9, color="#2563EB", line=dict(width=1, color="#1e40af")),
            name="故障データ",
            hovertemplate="t = %{x:.4g}<br>F = %{y:.4f}<extra></extra>",
        ))
    # Suspended data (rug)
    if len(t_cens) > 0:
        fig_prob.add_trace(go.Scatter(
            x=t_cens.tolist(), y=[0.01] * len(t_cens),
            mode="markers",
            marker=dict(symbol="line-ns", size=10, color="#DC2626", line=dict(width=2, color="#DC2626")),
            name="打ち切りデータ",
            hovertemplate="t = %{x:.4g} (打ち切り)<extra></extra>",
        ))
    # Fitted line
    F_line = wb.cdf(t_line)
    F_line_clipped = np.clip(F_line, 1e-15, 1 - 1e-15)
    fig_prob.add_trace(go.Scatter(
        x=t_line.tolist(), y=F_line_clipped.tolist(),
        mode="lines",
        line=dict(color="#16A34A", width=2),
        name="Weibull フィット",
        hovertemplate="t = %{x:.4g}<br>F = %{y:.4f}<extra></extra>",
    ))
    # CI band
    lo, hi = wb.cdf_ci(t_line, cl)
    lo = np.clip(lo, 1e-15, 1 - 1e-15)
    hi = np.clip(hi, 1e-15, 1 - 1e-15)
    fig_prob.add_trace(go.Scatter(
        x=np.concatenate([t_line, t_line[::-1]]).tolist(),
        y=np.concatenate([hi, lo[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(22,163,74,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(cl*100)}% 信頼区間",
        hoverinfo="skip",
    ))
    fig_prob.update_layout(
        **_plot_layout("ワイブル確率プロット", "時間 t", "累積故障率 F(t)"),
        xaxis=dict(type="log", title="時間 t"),
        yaxis=dict(type="linear", title="累積故障率 F(t)",
                   tickformat=".0%",
                   range=[0, 1]),
    )

    # ── 2. CDF / SF plot ────────────────────────────────────────
    fig_cdf = make_subplots(rows=1, cols=2,
                            subplot_titles=("累積故障率 CDF", "信頼性関数 SF"))
    F_line = wb.cdf(t_line)
    S_line = wb.sf(t_line)
    lo_F, hi_F = wb.cdf_ci(t_line, cl)

    fig_cdf.add_trace(go.Scatter(x=t_line.tolist(), y=F_line.tolist(),
                                 mode="lines", line=dict(color="#2563EB", width=2),
                                 name="CDF", legendgroup="cdf"), row=1, col=1)
    fig_cdf.add_trace(go.Scatter(
        x=np.concatenate([t_line, t_line[::-1]]).tolist(),
        y=np.concatenate([hi_F, lo_F[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(cl*100)}% CI", legendgroup="cdf_ci", showlegend=False,
        hoverinfo="skip"), row=1, col=1)
    fig_cdf.add_trace(go.Scatter(x=t_line.tolist(), y=S_line.tolist(),
                                 mode="lines", line=dict(color="#16A34A", width=2),
                                 name="SF", legendgroup="sf"), row=1, col=2)
    lo_S, hi_S = 1 - hi_F, 1 - lo_F
    fig_cdf.add_trace(go.Scatter(
        x=np.concatenate([t_line, t_line[::-1]]).tolist(),
        y=np.concatenate([hi_S, lo_S[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(22,163,74,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(cl*100)}% CI", legendgroup="sf_ci", showlegend=False,
        hoverinfo="skip"), row=1, col=2)
    fig_cdf.update_xaxes(title_text="時間 t", type="log")
    fig_cdf.update_yaxes(title_text="F(t)", tickformat=".0%", row=1, col=1)
    fig_cdf.update_yaxes(title_text="R(t)", tickformat=".0%", row=1, col=2)
    fig_cdf.update_layout(**_plot_layout("CDF / SF"))

    # ── 3. PDF + HF plot ────────────────────────────────────────
    fig_pdf = make_subplots(rows=1, cols=2,
                            subplot_titles=("確率密度関数 PDF", "故障率関数 HF"))
    P_line = wb.pdf(t_line)
    H_line = wb.hf(t_line)
    fig_pdf.add_trace(go.Scatter(x=t_line.tolist(), y=P_line.tolist(),
                                 mode="lines", line=dict(color="#7C3AED", width=2),
                                 name="PDF"), row=1, col=1)
    fig_pdf.add_trace(go.Scatter(x=t_line.tolist(), y=H_line.tolist(),
                                 mode="lines", line=dict(color="#DC2626", width=2),
                                 name="HF"), row=1, col=2)
    fig_pdf.update_xaxes(title_text="時間 t", type="log")
    fig_pdf.update_yaxes(title_text="f(t)", row=1, col=1)
    fig_pdf.update_yaxes(title_text="h(t)", row=1, col=2)
    fig_pdf.update_layout(**_plot_layout("PDF / HF"))

    return {
        "prob": fig_prob.to_json(),
        "cdf": fig_cdf.to_json(),
        "pdf": fig_pdf.to_json(),
    }


def _make_ds_plots(ds, times, censored, cl):
    """Return Plotly JSON strings for DS-Weibull plots."""
    t_fail, mr, x_emp, y_emp = weibull_plot_data(times, censored)
    t_cens = times[censored] if np.any(censored) else np.array([])

    t_min = max(times.min() * 0.1, 1e-6)
    t_max = times.max() * 3
    t_line = np.logspace(np.log10(t_min), np.log10(t_max), 300)

    F_line = ds.cdf(t_line)
    S_line = ds.sf(t_line)
    P_line = ds.pdf(t_line)
    H_line = ds.hf(t_line)
    lo_F, hi_F = ds.cdf_ci(t_line, cl)

    # ── 1. CDF / probability plot ───────────────────────────────
    fig_prob = go.Figure()
    if len(t_fail) > 0:
        fig_prob.add_trace(go.Scatter(
            x=t_fail.tolist(), y=mr.tolist(),
            mode="markers",
            marker=dict(symbol="circle", size=9, color="#2563EB",
                        line=dict(width=1, color="#1e40af")),
            name="故障データ",
        ))
    if len(t_cens) > 0:
        fig_prob.add_trace(go.Scatter(
            x=t_cens.tolist(), y=[0.01] * len(t_cens),
            mode="markers",
            marker=dict(symbol="line-ns", size=10, color="#DC2626",
                        line=dict(width=2, color="#DC2626")),
            name="打ち切りデータ",
        ))
    fig_prob.add_trace(go.Scatter(
        x=t_line.tolist(), y=F_line.tolist(),
        mode="lines", line=dict(color="#D97706", width=2),
        name="DS-Weibull フィット",
    ))
    fig_prob.add_trace(go.Scatter(
        x=np.concatenate([t_line, t_line[::-1]]).tolist(),
        y=np.concatenate([hi_F, lo_F[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(217,119,6,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(cl*100)}% 信頼区間", hoverinfo="skip",
    ))
    # DS asymptote
    fig_prob.add_hline(
        y=float(ds.DS), line_dash="dash", line_color="#64748B",
        annotation_text=f"DS = {ds.DS*100:.1f}%",
        annotation_position="top right",
    )
    fig_prob.update_layout(
        **_plot_layout("DS-Weibull CDF", "時間 t", "累積故障率 F(t)"),
        xaxis=dict(type="log", title="時間 t"),
        yaxis=dict(title="累積故障率 F(t)", tickformat=".0%", range=[0, min(ds.DS * 1.3, 1)]),
    )

    # ── 2. CDF / SF ─────────────────────────────────────────────
    fig_cdf = make_subplots(rows=1, cols=2, subplot_titles=("累積故障率 CDF", "信頼性関数 SF"))
    fig_cdf.add_trace(go.Scatter(x=t_line.tolist(), y=F_line.tolist(),
                                 mode="lines", line=dict(color="#D97706", width=2), name="CDF",
                                 legendgroup="cdf"), row=1, col=1)
    fig_cdf.add_trace(go.Scatter(
        x=np.concatenate([t_line, t_line[::-1]]).tolist(),
        y=np.concatenate([hi_F, lo_F[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(217,119,6,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(cl*100)}% CI", legendgroup="cdf_ci", showlegend=False, hoverinfo="skip"),
        row=1, col=1)
    fig_cdf.add_trace(go.Scatter(x=t_line.tolist(), y=S_line.tolist(),
                                 mode="lines", line=dict(color="#0891B2", width=2), name="SF",
                                 legendgroup="sf"), row=1, col=2)
    lo_S, hi_S = 1 - hi_F, 1 - lo_F
    fig_cdf.add_trace(go.Scatter(
        x=np.concatenate([t_line, t_line[::-1]]).tolist(),
        y=np.concatenate([hi_S, lo_S[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(8,145,178,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(cl*100)}% CI", legendgroup="sf_ci", showlegend=False, hoverinfo="skip"),
        row=1, col=2)
    fig_cdf.update_xaxes(title_text="時間 t", type="log")
    fig_cdf.update_yaxes(title_text="F(t)", tickformat=".0%", row=1, col=1)
    fig_cdf.update_yaxes(title_text="R(t)", tickformat=".0%", row=1, col=2)
    fig_cdf.update_layout(**_plot_layout("CDF / SF (DS-Weibull)"))

    # ── 3. PDF + HF ─────────────────────────────────────────────
    fig_pdf = make_subplots(rows=1, cols=2, subplot_titles=("確率密度関数 PDF", "故障率関数 HF"))
    fig_pdf.add_trace(go.Scatter(x=t_line.tolist(), y=P_line.tolist(),
                                 mode="lines", line=dict(color="#7C3AED", width=2), name="PDF"),
                      row=1, col=1)
    fig_pdf.add_trace(go.Scatter(x=t_line.tolist(), y=H_line.tolist(),
                                 mode="lines", line=dict(color="#DC2626", width=2), name="HF"),
                      row=1, col=2)
    fig_pdf.update_xaxes(title_text="時間 t", type="log")
    fig_pdf.update_yaxes(title_text="f(t)", row=1, col=1)
    fig_pdf.update_yaxes(title_text="h(t)", row=1, col=2)
    fig_pdf.update_layout(**_plot_layout("PDF / HF (DS-Weibull)"))

    return {
        "prob": fig_prob.to_json(),
        "cdf": fig_cdf.to_json(),
        "pdf": fig_pdf.to_json(),
    }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _fmt(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}g}"


def _plot_layout(title="", xaxis_title="", yaxis_title=""):
    return dict(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=30, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
