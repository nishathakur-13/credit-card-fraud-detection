import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Sleek Modern Dark Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI - Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Neon Purple Dark Dashboard UI
st.markdown("""
    <style>
    /* Dark Theme Backgrounds */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #11151F;
        border-right: 1px solid #1E2638;
    }
    
    /* Card Container Styling */
    .stat-card {
        background-color: #131824;
        border: 1px solid #1E2638;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    .stat-title {
        color: #8A99AD;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .stat-value {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 700;
        margin: 6px 0;
    }
    .stat-change-up {
        color: #10B981;
        font-size: 12px;
        font-weight: 600;
    }
    .stat-change-down {
        color: #EF4444;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Alert Cards */
    .alert-card {
        background-color: #131824;
        border: 1px solid #1E2638;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .alert-title {
        color: #FFFFFF;
        font-size: 14px;
        font-weight: 600;
    }
    .alert-sub {
        color: #8A99AD;
        font-size: 12px;
    }
    
    /* Primary Purple Button Override */
    div.stButton > button:first-child {
        background-color: #6366F1 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    div.stButton > button:first-child:hover {
        background-color: #4F46E5 !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Sidebar Navigation & Admin Profile
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="background: #6366F1; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">🛡️</div>
            <div>
                <h3 style="margin: 0; padding: 0; font-size: 18px; color: white;">FraudShield AI</h3>
                <p style="margin: 0; padding: 0; font-size: 11px; color: #8A99AD;">Credit Card Fraud Detection</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    nav_option = st.radio(
        "Navigation", 
        ["📊 Dashboard", "💳 Transactions", "🚨 Alerts", "📈 Analytics", "🤖 Models", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ⚙️ Live Tester Controls")
    amount = st.number_input("Transaction Amount (₹)", min_value=0.0, max_value=500000.0, value=12450.0, step=500.0)
    location = st.selectbox("Transaction Location", ["New Delhi, IN", "Mumbai, IN", "Moscow, RU (High Risk)", "London, UK", "New York, US"])
    
    st.divider()
    analyze_btn = st.button("⚡ Test Fraud Model", use_container_width=True)

    st.markdown("""
        <br/><br/>
        <div style="display: flex; align-items: center; gap: 10px; background: #131824; padding: 12px; border-radius: 10px; border: 1px solid #1E2638;">
            <div style="background: #6366F1; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; color: white;">NK</div>
            <div>
                <p style="margin:0; font-size: 13px; font-weight: 600; color: white;">Admin User</p>
                <p style="margin:0; font-size: 11px; color: #8A99AD;">Administrator</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Main Dashboard Body
# -----------------------------------------------------------------------------
if "Dashboard" in nav_option:
    # Header Bar
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.title("Dashboard")
        st.caption("Real-time overview of credit card fraud detection system.")
    with h_col2:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.button("📥 Export Report", use_container_width=True)

    st.divider()

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Total Transactions</div>
                <div class="stat-value">12,568</div>
                <div class="stat-change-up">▲ +18.2% <span style="color: #8A99AD;">from last week</span></div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Fraudulent Transactions</div>
                <div class="stat-value" style="color: #EF4444;">156</div>
                <div class="stat-change-down">▲ +7.1% <span style="color: #8A99AD;">from last week</span></div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Fraud Detection Rate</div>
                <div class="stat-value" style="color: #10B981;">96.7%</div>
                <div class="stat-change-up">▲ +2.3% <span style="color: #8A99AD;">from last week</span></div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Money Saved</div>
                <div class="stat-value" style="color: #6366F1;">₹8,24,765</div>
                <div class="stat-change-up">▲ +12.5% <span style="color: #8A99AD;">from last week</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Main Visualizations Row
    chart_col, donut_col = st.columns([2, 1], gap="large")

    with chart_col:
        st.subheader("Transaction Overview")
        
        # Line Chart for Transactions vs Fraud
        dates = ['May 24', 'May 25', 'May 26', 'May 27', 'May 28', 'May 29', 'May 30']
        total_tx = [1500, 2200, 2400, 2900, 3800, 3100, 3500]
        fraud_tx = [40, 80, 50, 90, 140, 110, 156]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=dates, y=total_tx, mode='lines+markers', name='Total Transactions', line=dict(color='#6366F1', width=3), fill='tonexty', fillcolor='rgba(99, 102, 241, 0.1)'))
        fig_line.add_trace(go.Scatter(x=dates, y=fraud_tx, mode='lines+markers', name='Fraudulent Transactions', line=dict(color='#EF4444', width=3)))
        
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#8A99AD'},
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_line.update_xaxes(showgrid=False)
        fig_line.update_yaxes(showgrid=True, gridcolor='#1E2638')
        
        st.plotly_chart(fig_line, use_container_width=True)

    with donut_col:
        st.subheader("Fraud by Category")
        
        donut_labels = ['Stolen Card', 'Online Fraud', 'Card Not Present', 'Identity Theft', 'Other']
        donut_values = [38, 28, 20, 10, 4]
        donut_colors = ['#6366F1', '#3B82F6', '#06B6D4', '#10B981', '#F59E0B']

        fig_donut = go.Figure(data=[go.Pie(
            labels=donut_labels, 
            values=donut_values, 
            hole=.6,
            marker_colors=donut_colors
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # Bottom Row: Alerts, Model Performance Gauge
    bot_col1, bot_col2 = st.columns([1, 1], gap="large")

    with bot_col1:
        st.subheader("Recent Alerts")
        
        st.markdown("""
            <div class="alert-card">
                <div class="alert-title">🚨 High Risk Transaction Detected</div>
                <div class="alert-sub">TXN100081 • ₹2,45,000.00 • New York, USA • 2 min ago</div>
            </div>
            <div class="alert-card" style="border-left-color: #F59E0B;">
                <div class="alert-title" style="color: #F59E0B;">⚠️ Multiple Failed Attempts</div>
                <div class="alert-sub">TXN100082 • 5 Attempts • London, UK • 15 min ago</div>
            </div>
            <div class="alert-card">
                <div class="alert-title">🚨 Suspicious Location Anomaly</div>
                <div class="alert-sub">TXN100083 • ₹1,12,000.00 • Moscow, Russia • 30 min ago</div>
            </div>
        """, unsafe_allow_html=True)

    with bot_col2:
        st.subheader("Model Performance")
        
        # Semi-Circle Accuracy Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=96.7,
            number={'suffix': "%", 'font': {'size': 36, 'color': "white"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#6366F1"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [50, 85], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=220, 
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

elif "Transactions" in nav_option:
    st.title("Transactions Registry")
    st.caption("Manage and review all processed transactions.")
    st.divider()
    
    # Mock Transactions Table
    tx_data = pd.DataFrame({
        "Transaction ID": ["TXN100081", "TXN100082", "TXN100083", "TXN100084", "TXN100085"],
        "Card Number": ["**** **** 4587", "**** **** 7890", "**** **** 1234", "**** **** 9876", "**** **** 9012"],
        "Amount (INR)": ["₹2,45,000.00", "₹85,000.00", "₹1,12,000.00", "₹7,500.00", "₹3,70,000.00"],
        "Location": ["New York, USA", "London, UK", "Moscow, Russia", "Paris, France", "Tokyo, Japan"],
        "Risk Level": ["High", "Medium", "High", "Low", "Approved"],
        "Status": ["Blocked", "Flagged", "Blocked", "Approved", "Approved"]
    })
    st.dataframe(tx_data, use_container_width=True)

else:
    st.title(nav_option)
    st.info("Module interface active. Select **Dashboard** in the sidebar to return to main metrics.")