import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

st.set_page_config(
    page_title="Electronics Inventory Control",
    page_icon="▢",
    layout="wide",
)

# ---------- THEME ----------
PRIMARY = "#0B1E33"      # deep navy
ACCENT = "#FF8A00"       # amber/warning
ACCENT2 = "#3DDC97"      # signal green
GRID = "#1E3A5C"
TEXT = "#E8EEF5"
MUTED = "#7C93AD"
BG = "#08141F"

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; color: {TEXT}; }}
h1, h2, h3 {{ font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.5px; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; }}
[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }}
.stSlider label, .stSelectbox label {{ color: {MUTED}; }}
hr {{ border-color: {GRID}; }}
.block-container {{ padding-top: 2rem; }}
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    elec = df[df["Category"] == "Electronics"].copy()
    elec["Date"] = pd.to_datetime(elec["Date"])
    return elec

df = load_data()

# ---------- HEADER ----------
st.markdown("### ▢ ELECTRONICS INVENTORY CONTROL")
st.markdown(
    f"<p style='color:{MUTED}; margin-top:-10px;'>Reorder planning, demand forecasting, and stockout risk "
    f"across {df['Store ID'].nunique()} stores · {df['Product ID'].nunique()} SKUs · "
    f"{df['Date'].min().date()} – {df['Date'].max().date()}</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.markdown("### PARAMETERS")
stores = sorted(df["Store ID"].unique())
products = sorted(df["Product ID"].unique())

sel_store = st.sidebar.selectbox("Store", ["All"] + stores)
sel_product = st.sidebar.selectbox("SKU", ["All"] + products)

st.sidebar.markdown("---")
st.sidebar.markdown("##### Reorder model inputs")
lead_time = st.sidebar.slider("Lead time (days)", 1, 30, 7)
service_level = st.sidebar.slider("Target service level (%)", 80, 99, 95)
review_period = st.sidebar.slider("Review period (days)", 1, 14, 1)

z = norm.ppf(service_level / 100)

# ---------- FILTER ----------
fdf = df.copy()
if sel_store != "All":
    fdf = fdf[fdf["Store ID"] == sel_store]
if sel_product != "All":
    fdf = fdf[fdf["Product ID"] == sel_product]

fdf = fdf.sort_values("Date")

# ---------- CORE CALCS ----------
daily = fdf.groupby("Date").agg(
    units_sold=("Units Sold", "sum"),
    inventory=("Inventory Level", "mean"),
    demand_forecast=("Demand Forecast", "sum"),
    units_ordered=("Units Ordered", "sum"),
).reset_index()

avg_demand = daily["units_sold"].mean()
std_demand = daily["units_sold"].std()

safety_stock = z * std_demand * np.sqrt(lead_time)
reorder_point = (avg_demand * lead_time) + safety_stock
order_up_to = (avg_demand * (lead_time + review_period)) + safety_stock

# Stockout risk: days where inventory level fell below that day's demand
fdf["stockout_flag"] = (fdf["Inventory Level"] < fdf["Units Sold"]).astype(int)
stockout_rate = fdf["stockout_flag"].mean() * 100

# Days inventory would currently last vs reorder point
avg_inventory = daily["inventory"].mean()
days_of_supply = avg_inventory / avg_demand if avg_demand > 0 else 0

# ---------- TOP METRICS ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Daily Demand", f"{avg_demand:,.1f}", help="Mean units sold/day across selection")
c2.metric("Demand Std Dev", f"{std_demand:,.1f}", help="Day-to-day demand variability")
c3.metric("Safety Stock", f"{safety_stock:,.0f} units")
c4.metric("Reorder Point", f"{reorder_point:,.0f} units")
c5.metric("Stockout Rate", f"{stockout_rate:.1f}%",
          delta=f"{'High' if stockout_rate > 15 else 'OK'}",
          delta_color="inverse")

st.markdown("---")

# ---------- ROW: INVENTORY VS REORDER POINT ----------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("##### Inventory level vs. reorder point")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["inventory"],
        mode="lines", name="Inventory Level",
        line=dict(color=ACCENT2, width=1.5)
    ))
    fig.add_hline(y=reorder_point, line_dash="dash", line_color=ACCENT,
                   annotation_text="Reorder Point", annotation_position="top left")
    fig.add_hline(y=safety_stock, line_dash="dot", line_color=MUTED,
                   annotation_text="Safety Stock", annotation_position="bottom left")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10), height=350,
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="Units"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("##### Reorder model breakdown")
    st.markdown(f"""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 0.9rem; line-height: 1.9;'>
    <span style='color:{MUTED}'>Z-score @ {service_level}% SL</span><br>
    <span style='color:{ACCENT}; font-size:1.3rem'>{z:.2f}</span><br><br>
    <span style='color:{MUTED}'>Demand during lead time</span><br>
    <span style='color:{TEXT}; font-size:1.3rem'>{avg_demand * lead_time:,.0f} units</span><br><br>
    <span style='color:{MUTED}'>+ Safety stock buffer</span><br>
    <span style='color:{ACCENT}; font-size:1.3rem'>{safety_stock:,.0f} units</span><br><br>
    <span style='color:{MUTED}'>= Reorder point</span><br>
    <span style='color:{ACCENT2}; font-size:1.5rem'>{reorder_point:,.0f} units</span><br><br>
    <span style='color:{MUTED}'>Order-up-to level</span><br>
    <span style='color:{TEXT}; font-size:1.3rem'>{order_up_to:,.0f} units</span><br><br>
    <span style='color:{MUTED}'>Days of supply (avg)</span><br>
    <span style='color:{TEXT}; font-size:1.3rem'>{days_of_supply:.1f} days</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- ROW: DEMAND FORECAST ACCURACY + SEASONALITY ----------
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("##### Demand: actual vs. forecast")
    weekly = daily.set_index("Date").resample("W").sum().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=weekly["Date"], y=weekly["units_sold"],
                               name="Actual", line=dict(color=ACCENT2, width=2)))
    fig2.add_trace(go.Scatter(x=weekly["Date"], y=weekly["demand_forecast"],
                               name="Forecast", line=dict(color=ACCENT, width=2, dash="dot")))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="Units/week"),
        legend=dict(orientation="h", y=1.15)
    )
    st.plotly_chart(fig2, use_container_width=True)

    mape = (np.abs(daily["units_sold"] - daily["demand_forecast"]) / daily["units_sold"].replace(0, np.nan)).mean() * 100
    st.markdown(f"<span style='color:{MUTED}'>Mean Absolute % Error (MAPE): "
                f"<span style='color:{ACCENT}; font-family:IBM Plex Mono'>{mape:.1f}%</span></span>",
                unsafe_allow_html=True)

with col_b:
    st.markdown("##### Stockout risk by store")
    risk_by_store = df.groupby("Store ID").apply(
        lambda x: (x["Inventory Level"] < x["Units Sold"]).mean() * 100
    ).reset_index(name="stockout_rate")
    fig3 = px.bar(risk_by_store, x="Store ID", y="stockout_rate",
                   color="stockout_rate", color_continuous_scale=["#3DDC97", "#FF8A00", "#FF4D4D"])
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="Stockout rate (%)"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ---------- ROW: REGION + PROMO IMPACT ----------
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("##### Avg daily demand by region")
    region_demand = df.groupby("Region")["Units Sold"].mean().reset_index()
    fig4 = px.bar(region_demand, x="Region", y="Units Sold", color="Units Sold",
                   color_continuous_scale=["#1E3A5C", ACCENT2])
    fig4.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="Avg units sold/day"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    st.markdown("##### Promotion impact on demand")
    promo = df.groupby("Holiday/Promotion")["Units Sold"].mean().reset_index()
    promo["Holiday/Promotion"] = promo["Holiday/Promotion"].map({0: "Normal day", 1: "Holiday/Promo"})
    fig5 = px.bar(promo, x="Holiday/Promotion", y="Units Sold", color="Holiday/Promotion",
                   color_discrete_map={"Normal day": "#1E3A5C", "Holiday/Promo": ACCENT})
    fig5.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        xaxis=dict(gridcolor=GRID, title=""), yaxis=dict(gridcolor=GRID, title="Avg units sold/day"),
        showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.markdown(
    f"<p style='color:{MUTED}; font-size:0.8rem;'>Reorder point = (avg daily demand × lead time) + safety stock. "
    f"Safety stock = z(service level) × σ(demand) × √(lead time). "
    f"Data: synthetic retail inventory dataset, Electronics category.</p>",
    unsafe_allow_html=True
)
