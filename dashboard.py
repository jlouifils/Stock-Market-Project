# dashboard.py
import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from state_store import (
    init_db, get_engine_state, get_latest_universe,
    get_latest_risk, get_risk_series, get_positions,
    get_orders, get_signals, get_decisions
)

PORT = int(os.getenv("DASH_PORT", "8050"))

init_db()
app = Dash(__name__)
app.title = "Trading Dashboard"


def badge(text, color):
    return html.Span(text, style={
        "display": "inline-block",
        "padding": "6px 10px",
        "borderRadius": "999px",
        "background": color,
        "color": "white",
        "fontWeight": "700",
        "marginRight": "8px",
        "fontSize": "12px"
    })


def card(title, children):
    return html.Div([
        html.Div(title, style={"fontWeight": "800", "marginBottom": "8px"}),
        html.Div(children)
    ], style={
        "background": "#111827",
        "color": "#E5E7EB",
        "borderRadius": "14px",
        "padding": "14px",
        "boxShadow": "0 6px 18px rgba(0,0,0,0.25)"
    })


def table_from_rows(rows):
    if not rows:
        return html.Div("No data yet.")
    df = pd.DataFrame(rows)
    return html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th(c) for c in df.columns])),
            html.Tbody([
                html.Tr([html.Td(str(df.iloc[i][c])) for c in df.columns])
                for i in range(min(len(df), 25))
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ], style={"overflowX": "auto"})


app.layout = html.Div([
    html.Div([
        html.Div("Alpaca-style Trading Dashboard", style={
            "fontSize": "20px", "fontWeight": "900", "color": "#111827"
        }),
        html.Div(id="badges", style={"marginTop": "10px"})
    ], style={"padding": "16px"}),

    html.Div([
        html.Div(id="top_cards", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "12px",
            "padding": "0 16px 16px 16px"
        }),
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id="equity_chart", config={"displayModeBar": False})
        ], style={"background": "white", "borderRadius": "14px", "padding": "12px", "margin": "0 16px"}),

        html.Div([
            html.Div([
                html.H3("Positions", style={"margin": "0 0 8px 0"}),
                html.Div(id="positions_tbl")
            ], style={"background": "white", "borderRadius": "14px", "padding": "12px", "margin": "12px 16px"}),

            html.Div([
                html.H3("Orders", style={"margin": "0 0 8px 0"}),
                html.Div(id="orders_tbl")
            ], style={"background": "white", "borderRadius": "14px", "padding": "12px", "margin": "12px 16px"}),

            html.Div([
                html.H3("Signals", style={"margin": "0 0 8px 0"}),
                html.Div(id="signals_tbl")
            ], style={"background": "white", "borderRadius": "14px", "padding": "12px", "margin": "12px 16px"}),

            html.Div([
                html.H3("Decisions (latest)", style={"margin": "0 0 8px 0"}),
                html.Div(id="decisions_tbl")
            ], style={"background": "white", "borderRadius": "14px", "padding": "12px", "margin": "12px 16px"}),
        ])
    ]),

    dcc.Interval(id="tick", interval=1500, n_intervals=0)
], style={"fontFamily": "system-ui, Segoe UI, Arial", "background": "#F3F4F6"})


@app.callback(
    Output("badges", "children"),
    Output("top_cards", "children"),
    Output("equity_chart", "figure"),
    Output("positions_tbl", "children"),
    Output("orders_tbl", "children"),
    Output("signals_tbl", "children"),
    Output("decisions_tbl", "children"),
    Input("tick", "n_intervals"),
)
def refresh(_):
    eng = get_engine_state() or {}
    uni = get_latest_universe() or {}
    risk = get_latest_risk() or {}
    series = get_risk_series(400)

    trading_on = bool(eng.get("trading_enabled", 0))
    kill = bool(eng.get("kill_switch", 0))

    badges = []
    badges.append(badge(f"TRADING {'ON' if trading_on else 'OFF'}", "#059669" if trading_on else "#6B7280"))
    badges.append(badge(f"KILL {'TRIPPED' if kill else 'OK'}", "#DC2626" if kill else "#2563EB"))
    if uni.get("symbols"):
        badges.append(badge(f"UNIVERSE {len(uni['symbols'].split(','))}", "#111827"))
    if eng.get("last_resync_ts"):
        badges.append(badge(f"RESYNC {eng['last_resync_ts']}", "#7C3AED"))
    if eng.get("last_enforce_action"):
        badges.append(badge(f"ENFORCE {eng['last_enforce_action']}", "#F59E0B"))

    # Top cards
    equity = float(risk.get("equity", 0.0) or 0.0)
    cash = float(risk.get("cash", 0.0) or 0.0)
    bp = float(risk.get("buying_power", 0.0) or 0.0)
    dd = float(risk.get("drawdown_pct", 0.0) or 0.0)
    gross = float(risk.get("gross_exposure", 0.0) or 0.0)
    invested = float(risk.get("invested_pct", 0.0) or 0.0)

    cards = [
        card("Equity", f"${equity:,.2f}"),
        card("Cash", f"${cash:,.2f}"),
        card("Buying Power", f"${bp:,.2f}"),
        card("Drawdown", f"{dd*100:,.2f}%"),
    ]

    # Equity chart
    if series:
        df = pd.DataFrame(series)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df["equity"], mode="lines", name="Equity"))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
        )
    else:
        fig = go.Figure()
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))

    positions_tbl = table_from_rows(get_positions())
    orders_tbl = table_from_rows(get_orders(120))
    signals_tbl = table_from_rows(get_signals())
    decisions_tbl = table_from_rows(get_decisions(120))

    return badges, cards, fig, positions_tbl, orders_tbl, signals_tbl, decisions_tbl


def run_dashboard():
    app.run(host="127.0.0.1", port=PORT, debug=False)


if __name__ == "__main__":
    run_dashboard()
