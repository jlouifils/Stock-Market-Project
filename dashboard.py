import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import alpaca_trade_api as tradeapi
import os
from datetime import datetime

# ==============================
# Alpaca Connection
# ==============================
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ==============================
# Dash App
# ==============================
app = dash.Dash(__name__)
app.title = "Live Trading Dashboard"

# ==============================
# Layout
# ==============================
app.layout = html.Div(
    style={"backgroundColor": "#0e0e0e", "color": "white", "padding": "20px"},
    children=[
        html.H2("📈 Live Trading Dashboard"),

        html.Div(id="account-summary", style={"marginBottom": "20px"}),

        dcc.Graph(id="equity-curve"),

        html.H3("Manual Trading"),
        html.Div([
            dcc.Input(id="trade-symbol", placeholder="Symbol", value="AAPL"),
            dcc.Input(id="trade-qty", placeholder="Qty", type="number", value=10),
            html.Button("BUY", id="buy-btn", n_clicks=0),
            html.Button("SELL", id="sell-btn", n_clicks=0),
            html.Div(id="trade-status")
        ], style={"marginBottom": "30px"}),

        html.H3("Open Positions"),
        dash_table.DataTable(
            id="positions-table",
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#111", "color": "white"},
            style_header={"backgroundColor": "#222", "fontWeight": "bold"},
        ),

        html.H3("Recent Orders"),
        dash_table.DataTable(
            id="orders-table",
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#111", "color": "white"},
            style_header={"backgroundColor": "#222", "fontWeight": "bold"},
        ),

        dcc.Interval(id="refresh", interval=10_000, n_intervals=0)
    ]
)

# ==============================
# Callbacks
# ==============================

@app.callback(
    Output("account-summary", "children"),
    Output("equity-curve", "figure"),
    Output("positions-table", "data"),
    Output("positions-table", "columns"),
    Output("orders-table", "data"),
    Output("orders-table", "columns"),
    Input("refresh", "n_intervals"),
)
def update_dashboard(_):
    account = alpaca.get_account()

    # ----- Equity Curve -----
    history = alpaca.get_portfolio_history(period="1D", timeframe="5Min")
    equity_df = pd.DataFrame({
        "time": pd.to_datetime(history.timestamp, unit="s"),
        "equity": history.equity
    })

    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(
        x=equity_df["time"],
        y=equity_df["equity"],
        mode="lines",
        name="Equity"
    ))
    equity_fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    # ----- Positions -----
    positions = alpaca.list_positions()
    pos_rows = []
    for p in positions:
        pos_rows.append({
            "Symbol": p.symbol,
            "Qty": p.qty,
            "Entry": float(p.avg_entry_price),
            "Price": float(p.current_price),
            "Market Value": float(p.market_value),
            "P/L": float(p.unrealized_pl)
        })

    pos_cols = [{"name": c, "id": c} for c in pos_rows[0].keys()] if pos_rows else []

    # ----- Orders -----
    orders = alpaca.list_orders(limit=20, status="all")
    order_rows = []
    for o in orders:
        order_rows.append({
            "Symbol": o.symbol,
            "Side": o.side,
            "Qty": o.qty,
            "Filled Avg": o.filled_avg_price,
            "Status": o.status,
            "Time": o.submitted_at.strftime("%Y-%m-%d %H:%M:%S")
        })

    order_cols = [{"name": c, "id": c} for c in order_rows[0].keys()] if order_rows else []

    summary = html.Div([
        html.B(f"Equity: ${float(account.equity):,.2f}"),
        html.Span(" | "),
        html.B(f"Cash: ${float(account.cash):,.2f}")
    ])

    return summary, equity_fig, pos_rows, pos_cols, order_rows, order_cols


@app.callback(
    Output("trade-status", "children"),
    Input("buy-btn", "n_clicks"),
    Input("sell-btn", "n_clicks"),
    State("trade-symbol", "value"),
    State("trade-qty", "value"),
)
def trade(buy, sell, symbol, qty):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    side = "buy" if ctx.triggered_id == "buy-btn" else "sell"

    try:
        alpaca.submit_order(
            symbol=symbol.upper(),
            qty=int(qty),
            side=side,
            type="market",
            time_in_force="day"
        )
        return f"Order submitted: {side.upper()} {qty} {symbol}"
    except Exception as e:
        return str(e)


# ==============================
# Run
# ==============================
def run_dashboard():
    app.run(debug=False)


if __name__ == "__main__":
    run_dashboard()
