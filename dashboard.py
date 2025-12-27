import logging
import time
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - DASHBOARD - %(levelname)s - %(message)s"
)

# ---------------- DASH APP ----------------
def create_app():
    app = Dash(__name__)
    app.title = "Trading Dashboard"

    app.layout = html.Div(
        style={"backgroundColor": "#0e1117", "color": "white", "padding": "20px"},
        children=[
            html.H1("📊 Live Trading Dashboard"),
            html.Div(id="status-text"),
            dcc.Graph(id="price-chart"),
            dcc.Interval(id="refresh", interval=2_000, n_intervals=0),
        ]
    )

    @app.callback(
        Output("status-text", "children"),
        Output("price-chart", "figure"),
        Input("refresh", "n_intervals"),
    )
    def update_dashboard(_):
        """
        This is a placeholder data source.
        In Phase 4.5 we will replace this with:
        - shared state
        - Redis
        - or SQLite
        """

        # Dummy data for now
        df = pd.DataFrame({
            "time": pd.date_range(end=pd.Timestamp.now(), periods=30),
            "price": pd.Series(range(30)).rolling(3).mean()
        })

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["price"],
                mode="lines",
                name="Price"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
        )

        return "🟢 Dashboard Running", fig

    return app


# ---------------- RUNNER ----------------
def run_dashboard():
    logging.info("🚀 Starting dashboard server")
    app = create_app()

    # BLOCKING CALL — THIS IS THE KEY FIX
    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050,
    )


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        logging.info("🛑 Dashboard stopped by user")
    except Exception as e:
        logging.exception(f"🔥 Dashboard crashed: {e}")
        raise
