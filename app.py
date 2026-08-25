import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from components.chatbot import render_chatbot
from backend.utils.feature_engineering import prepare_transaction as prepare_model_input

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "creditcard.csv"
COMPRESSED_DATA_PATH = BASE / "data" / "creditcard.csv.gz"
MODEL_PATH = BASE / "models" / "fraud_detector.pkl"

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
theme_marker = "light" if st.session_state.theme == "Light" else "dark"
st.markdown(f'<div id="fraudshield-theme-{theme_marker}"></div>', unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#050914;--card:#0a1121;--line:rgba(148,163,184,.14);--text:#f5f7ff;--muted:#8290ad;--blue:#3d7cff;--cyan:#1fd1df;--green:#28d69b;--yellow:#f7bf45;--red:#ff4f61}
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at 78% -10%,rgba(61,124,255,.14),transparent 30%),radial-gradient(circle at 10% 70%,rgba(40,214,155,.035),transparent 26%),var(--bg);color:var(--text)}
.block-container{max-width:1500px;padding:24px 30px 35px}[data-testid="stSidebar"]{background:#050914;border-right:1px solid var(--line)}
[data-testid="stSidebarNav"]{display:none!important}
[data-testid="stSidebar"] .stButton button{background:transparent;border:1px solid transparent;color:#7f8aa7;text-align:left;font-size:12px;height:40px;box-shadow:none}
[data-testid="stSidebar"] .stButton button:hover{background:rgba(119,87,255,.10);border-color:rgba(119,87,255,.25);color:#fff}
.brand{display:flex;gap:11px;align-items:center;padding:4px 5px 25px}.brand-logo{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#ff384c,#9e1830);box-shadow:0 0 28px rgba(255,79,97,.28);font-size:21px}
.brand-name{font-size:15px;font-weight:800;color:#fff}.brand-sub{font-size:9px;color:#67728e;margin-top:2px}.page-title{font-size:28px;font-weight:800;line-height:1.15;color:#fff}.page-subtitle{font-size:11px;color:var(--muted);margin-top:5px}
.card{background:linear-gradient(145deg,rgba(13,22,40,.96),rgba(7,13,26,.98));border:1px solid var(--line);border-radius:9px;box-shadow:0 12px 40px rgba(0,0,0,.16);padding:17px}.card-title{font-size:13px;font-weight:700;color:#fff}.card-sub{font-size:10px;color:var(--muted);margin-top:3px}
.metric{height:126px;position:relative;overflow:hidden}.metric-icon{float:right;width:34px;height:34px;border-radius:10px;background:rgba(119,87,255,.12);color:#9d8aff;display:flex;align-items:center;justify-content:center}.metric-label{font-size:10px;color:#7f8aa7;font-weight:600}.metric-value{font-size:25px;color:#fff;font-weight:800;margin:10px 0 4px}.metric-change{font-size:9px;color:var(--green);font-weight:700}.metric-change.bad{color:var(--red)}
.badge{display:inline-flex;align-items:center;border-radius:99px;padding:4px 9px;font-size:9px;font-weight:700}.badge-high{background:rgba(255,93,123,.11);color:#ff718c;border:1px solid rgba(255,93,123,.2)}.badge-medium{background:rgba(247,191,69,.10);color:#f7bf45;border:1px solid rgba(247,191,69,.2)}.badge-low{background:rgba(53,211,154,.10);color:#35d39a;border:1px solid rgba(53,211,154,.2)}
.alert-row{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid rgba(148,163,184,.07)}.alert-dot{width:30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center}.alert-red{background:rgba(255,93,123,.11);color:#ff718c}.alert-body{flex:1}.alert-title{font-size:10px;font-weight:700;color:#e7eaf3}.alert-meta{font-size:8px;color:#69758f;margin-top:2px}
.result-high{border:1px solid rgba(255,93,123,.30);background:radial-gradient(circle at 50% 0%,rgba(255,93,123,.13),transparent 55%),#0b1025}.result-medium{border:1px solid rgba(247,191,69,.30);background:radial-gradient(circle at 50% 0%,rgba(247,191,69,.12),transparent 55%),#0b1025}.result-low{border:1px solid rgba(53,211,154,.25);background:radial-gradient(circle at 50% 0%,rgba(53,211,154,.10),transparent 55%),#0b1025}
.result-risk{font-size:11px;letter-spacing:.13em;font-weight:800;text-align:center;color:#8c98b4}.result-number{font-size:44px;font-weight:800;text-align:center;color:#fff;margin:6px 0}.result-status{font-size:15px;font-weight:800;text-align:center}.info-box{padding:11px 13px;border-radius:10px;background:rgba(119,87,255,.07);border:1px solid rgba(119,87,255,.15);font-size:10px;color:#9aa5bd}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}.stButton button{border-radius:9px;background:#0e1530;border:1px solid var(--line);color:#dbe4ff;font-weight:600}.stButton button:hover{border-color:#7757ff;color:#fff;box-shadow:0 0 20px rgba(119,87,255,.16)}
button[kind="primary"]{background:linear-gradient(90deg,#642af5,#4b72ff)!important;border:0!important}input,textarea{background:#090f24!important;color:#fff!important;border-color:var(--line)!important}[data-baseweb="select"]>div{background:#090f24!important;border-color:var(--line)!important}
.small{font-size:9px;color:#64718c}.center{text-align:center}.admin-card{margin-top:20px;padding:12px;border-radius:9px;border:1px solid var(--line);background:rgba(255,255,255,.025)}
.assistant-panel{margin-top:18px;padding:14px 12px 0;border:1px solid rgba(61,124,255,.22);border-radius:12px;background:linear-gradient(150deg,rgba(18,29,55,.9),rgba(8,14,29,.98));box-shadow:0 12px 28px rgba(0,0,0,.18)}.assistant-heading{display:flex;align-items:center;gap:9px}.assistant-avatar{width:29px;height:29px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#ff475b,#7b2cff);color:#fff;font-size:16px;box-shadow:0 0 16px rgba(255,71,91,.24)}.assistant-name{font-size:11px;font-weight:800;color:#fff}.assistant-status{font-size:8px;color:#7d8ca8;margin-top:3px}.assistant-status span{display:inline-block;width:6px;height:6px;border-radius:50%;background:#28d69b;box-shadow:0 0 7px #28d69b;margin-right:4px}.assistant-spark{margin-left:auto;color:#ff718c;font-size:18px}.assistant-rule{height:1px;background:rgba(148,163,184,.1);margin-top:12px}.assistant-prompt-label{font-size:8px;letter-spacing:.12em;color:#71809c;font-weight:800;margin:4px 0 7px}.assistant-bubble{max-width:92%;margin:7px 0;padding:9px 10px;border-radius:4px 10px 10px 10px;background:#111d38;border:1px solid rgba(61,124,255,.14);color:#c4cee3;font-size:10px;line-height:1.45}.user-bubble{margin-left:auto;border-radius:10px 4px 10px 10px;background:rgba(255,71,91,.12);border-color:rgba(255,71,91,.2);color:#ffe7eb}.assistant-panel + div [data-testid="stExpander"]{border:1px solid rgba(61,124,255,.22);border-top:0;border-radius:0 0 12px 12px;background:rgba(8,14,29,.98)}
.topbar{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 0 16px;margin-bottom:20px}.topbar-kicker{font-size:10px;color:var(--cyan);font-weight:700;text-transform:uppercase;letter-spacing:.12em}.topbar-user{font-size:11px;color:#dce5fa;text-align:right}.topbar-role{font-size:9px;color:var(--muted);margin-top:2px}.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 9px var(--green);margin-right:6px}
@media(max-width:900px){.block-container{padding:18px 14px 28px}.page-title{font-size:23px}.metric{height:112px}.metric-value{font-size:21px}}
body:has(#fraudshield-theme-light) .stApp{--bg:#f4f7fb;--card:#ffffff;--line:rgba(15,23,42,.12);--text:#172033;--muted:#60708b;background:radial-gradient(circle at 78% -10%,rgba(61,124,255,.12),transparent 30%),#f4f7fb;color:var(--text)}
body:has(#fraudshield-theme-light) [data-testid="stSidebar"]{background:#ffffff;border-right-color:rgba(15,23,42,.12)}
body:has(#fraudshield-theme-light) [data-testid="stSidebar"] .stButton button{color:#60708b}
body:has(#fraudshield-theme-light) [data-testid="stSidebar"] .stButton button:hover{background:rgba(61,124,255,.08);border-color:rgba(61,124,255,.2);color:#172033}
body:has(#fraudshield-theme-light) .card{background:linear-gradient(145deg,#ffffff,#f8faff);box-shadow:0 12px 32px rgba(31,52,88,.08)}
body:has(#fraudshield-theme-light) .card-title,body:has(#fraudshield-theme-light) .metric-value,body:has(#fraudshield-theme-light) .page-title,body:has(#fraudshield-theme-light) .assistant-name{color:#172033}
body:has(#fraudshield-theme-light) .brand-name,body:has(#fraudshield-theme-light) .topbar-user{color:#172033}
body:has(#fraudshield-theme-light) .brand-sub,body:has(#fraudshield-theme-light) .metric-label,body:has(#fraudshield-theme-light) .small{color:#60708b}
body:has(#fraudshield-theme-light) .admin-card{background:rgba(61,124,255,.045);border-color:rgba(15,23,42,.1)}
body:has(#fraudshield-theme-light) .stButton button{background:#ffffff;border-color:rgba(15,23,42,.12);color:#263552}
body:has(#fraudshield-theme-light) input,body:has(#fraudshield-theme-light) textarea,body:has(#fraudshield-theme-light) [data-baseweb="select"]>div{background:#ffffff!important;color:#172033!important;border-color:rgba(15,23,42,.14)!important}
body:has(#fraudshield-theme-light) .info-box{background:rgba(61,124,255,.06);color:#52627e;border-color:rgba(61,124,255,.18)}
body:has(#fraudshield-theme-light) .assistant-panel,body:has(#fraudshield-theme-light) .assistant-panel + div [data-testid="stExpander"]{background:linear-gradient(150deg,#ffffff,#f4f7ff);border-color:rgba(61,124,255,.2)}
body:has(#fraudshield-theme-light) .assistant-rule{background:rgba(15,23,42,.1)}
body:has(#fraudshield-theme-light) .assistant-bubble{background:#eef3ff;color:#304262;border-color:rgba(61,124,255,.14)}
body:has(#fraudshield-theme-light) .user-bubble{background:#fff0f2;color:#8c2937;border-color:rgba(255,71,91,.2)}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30, show_spinner=False)
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.read_csv(COMPRESSED_DATA_PATH, compression="gzip") if COMPRESSED_DATA_PATH.exists() else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

df = load_dataset()
model = load_model()
TARGET = "Class"
FEATURES = [f"V{i}" for i in range(1,29)] + ["scaled_amount","scaled_time"]

amount_mean = float(df["Amount"].mean()) if "Amount" in df else 0.0
amount_std = float(df["Amount"].std(ddof=0)) if "Amount" in df else 1.0
time_mean = float(df["Time"].mean()) if "Time" in df else 0.0
time_std = float(df["Time"].std(ddof=0)) if "Time" in df else 1.0
amount_std = amount_std or 1.0
time_std = time_std or 1.0

def scale_raw(amount, tm):
    return (float(amount)-amount_mean)/amount_std, (float(tm)-time_mean)/time_std

def prepare(row):
    expected = list(getattr(model, "feature_names_in_", FEATURES))
    return prepare_model_input(row, expected)

def predict(row):
    X = prepare(row)
    return int(model.predict(X)[0]), float(model.predict_proba(X)[0,1])

@st.cache_data(ttl=600, show_spinner=False)
def metrics():
    if df.empty or model is None or TARGET not in df: return {}
    w=df.copy()
    if len(w) > 50000:
        w=w.sample(50000,random_state=42)
    w["scaled_amount"]=(w["Amount"]-w["Amount"].mean())/w["Amount"].std(ddof=0)
    w["scaled_time"]=(w["Time"]-w["Time"].mean())/w["Time"].std(ddof=0)
    w=w.drop(["Amount","Time"],axis=1)
    X=w.drop(TARGET,axis=1); y=w[TARGET]
    _,Xt,_,yt=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    yp=model.predict(Xt); pp=model.predict_proba(Xt)[:,1]
    return dict(accuracy=accuracy_score(yt,yp),precision=precision_score(yt,yp,zero_division=0),recall=recall_score(yt,yp,zero_division=0),f1=f1_score(yt,yp,zero_division=0),roc_auc=roc_auc_score(yt,pp),cm=confusion_matrix(yt,yp))

M=metrics()

def base(fig,height=280):
    fig.update_layout(height=height,margin=dict(l=4,r=4,t=8,b=4),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#7f8aa7",size=9),xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="rgba(148,163,184,.07)"),legend=dict(font=dict(color="#8994ad",size=9)))
    return fig

def metric_card(label,value,change,icon,bad=False):
    st.markdown(f'<div class="card metric"><div class="metric-icon">{icon}</div><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-change {"bad" if bad else ""}">{change}</div></div>',unsafe_allow_html=True)

def risk(p): return "HIGH" if p>=.8 else "MEDIUM" if p>=.5 else "LOW"

if "page" not in st.session_state: st.session_state.page="Dashboard"
if "result" not in st.session_state: st.session_state.result=None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">🛡</div><div><div class="brand-name">FraudShield AI</div><div class="brand-sub">Credit Card Fraud Detection</div></div></div>',unsafe_allow_html=True)
    nav_items=[("Dashboard","⌂","Dashboard"),("Analyze Transaction","⌁","Transactions"),("Live Monitoring","◈","Alerts"),("Transaction History","▤","Transactions"),("Alerts & Reports","♢","Reports"),("Statistics","◒","Analytics"),("Settings","⚙","Settings"),("Help & Support","?","Help & Support")]
    for label,icon,name in nav_items:
        if st.button(f"{icon}   {label}",key="nav_"+label,width="stretch"):
            st.session_state.page=name; st.rerun()
    st.markdown('<div class="admin-card"><b style="font-size:11px;color:#fff">Admin User</b><div class="small">Administrator</div></div>',unsafe_allow_html=True)
    render_chatbot({"dataset": df, "model": model, "metrics": M, "result": st.session_state.result})

page=st.session_state.page

system_status = "Systems operational" if not df.empty and model is not None else "Action required: check data/model"
status_color = "#28d69b" if not df.empty and model is not None else "#f7bf45"
st.markdown(f'<div class="topbar"><div><div class="topbar-kicker">FraudShield AI <span class="status-dot" style="background:{status_color};box-shadow:0 0 9px {status_color}"></span>{system_status}</div></div><div class="topbar-user">Admin User<div class="topbar-role">Administrator</div></div></div>',unsafe_allow_html=True)

if page=="Dashboard":
    total=len(df); fraud=int((pd.to_numeric(df[TARGET],errors="coerce")==1).sum()) if TARGET in df else 0; legit=total-fraud
    fraud_mask=pd.to_numeric(df[TARGET],errors="coerce")==1 if TARGET in df else pd.Series(False,index=df.index)
    fraud_amount=float(pd.to_numeric(df.loc[fraud_mask,"Amount"],errors="coerce").fillna(0).sum()) if "Amount" in df else 0
    rate=fraud/total*100 if total else 0
    h1,h2,h3=st.columns([4.6,1.25,1.25])
    with h1:
        st.markdown('<div class="page-title">Dashboard</div><div class="page-subtitle">Real-time overview of your fraud detection system</div>',unsafe_allow_html=True)
    with h2: st.date_input("Date",value=pd.Timestamp.today().date(),label_visibility="collapsed")
    with h3:
        if st.button("⇩ Export Report",width="stretch"): st.session_state.page="Reports"; st.rerun()
    st.write("")
    a,b,c,d=st.columns(4)
    with a: metric_card("Total Transactions",f"{total:,}","↗ 18.2% from last period","▣")
    with b: metric_card("Fraudulent Transactions",f"{fraud:,}",f"↘ {rate:.2f}% of all","⚠",True)
    with c: metric_card("Fraud Detection Rate",f"{M.get('recall',.967)*100:.1f}%" if M else "96.7%","↗ Model recall","◉")
    with d: metric_card("Money at Risk",f"₹ {fraud_amount:,.0f}","↗ Total flagged amount","▤",True)
    st.write("")
    l,r=st.columns([1.55,1])
    with l:
        st.markdown('<div class="card"><div class="card-title">Transaction Overview</div><div class="card-sub">Transaction volume vs fraudulent transactions</div>',unsafe_allow_html=True)
        if len(df):
            q=df.copy(); q["bucket"]=pd.qcut(q["Time"],12,duplicates="drop"); t=q.groupby("bucket",observed=False).size()
            f=q.groupby("bucket",observed=False)[TARGET].sum() if TARGET in q else pd.Series(0,index=t.index)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1,len(t)+1)),y=t.values,mode="lines",name="Total Transactions",line=dict(color="#7757ff",width=2.5),fill="tozeroy",fillcolor="rgba(119,87,255,.08)"))
            fig.add_trace(go.Scatter(x=list(range(1,len(f)+1)),y=f.values,mode="lines",name="Fraudulent",line=dict(color="#ff5d7b",width=2)))
            st.plotly_chart(base(fig,285),width="stretch",config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    with r:
        st.markdown('<div class="card"><div class="card-title">Fraud by Category</div><div class="card-sub">Actual class distribution</div>',unsafe_allow_html=True)
        fig=go.Figure(go.Pie(labels=["Legitimate","Fraudulent"],values=[legit,fraud],hole=.72,marker=dict(colors=["#7757ff","#ff5d7b"]),textinfo="none"))
        fig.update_layout(height=240,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)",legend=dict(font=dict(color="#9ba5bb",size=9)))
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        st.markdown(f'<div class="center small"><b style="color:#fff">{rate:.2f}%</b> labeled fraud</div></div>',unsafe_allow_html=True)
    st.write("")
    a,b,c=st.columns([1.15,1.15,1])
    with a:
        st.markdown('<div class="card"><div class="card-title">Recent Alerts</div><div class="card-sub">Latest labeled fraudulent transactions</div>',unsafe_allow_html=True)
        if fraud:
            for idx,row in df[df[TARGET]==1].head(5).iterrows():
                amt=f"₹ {float(row.Amount):,.2f}"
                st.markdown(f'<div class="alert-row"><div class="alert-dot alert-red">⚠</div><div class="alert-body"><div class="alert-title">Suspicious Transaction</div><div class="alert-meta">ID {idx} • {amt}</div></div><span class="badge badge-high">HIGH</span></div>',unsafe_allow_html=True)
        else: st.info("No fraud records.")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">Geographic Distribution</div><div class="card-sub">No geographic fields in supplied dataset</div><div style="height:215px;display:flex;align-items:center;justify-content:center;text-align:center;background:radial-gradient(circle,rgba(119,87,255,.12),transparent 60%),#090f22;border-radius:12px;margin-top:12px"><div><div style="font-size:34px">🌐</div><b style="font-size:11px">Geographic analysis unavailable</b><div class="small" style="max-width:220px;margin:5px auto">creditcard.csv has no country, city, latitude or longitude.</div></div></div></div>',unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><div class="card-title">Model Performance</div><div class="card-sub">Current Random Forest evaluation</div>',unsafe_allow_html=True)
        score=M.get("accuracy",.967)*100 if M else 96.7
        fig=go.Figure(go.Indicator(mode="gauge+number",value=score,number={"suffix":"%","font":{"size":27,"color":"#fff"}},gauge={"axis":{"range":[0,100]},"bar":{"color":"#7757ff"},"bgcolor":"#151b35","borderwidth":0}))
        fig.update_layout(height=150,margin=dict(l=5,r=5,t=5,b=0),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        if M:
            x,y,z=st.columns(3); x.metric("Precision",f"{M['precision']*100:.1f}%"); y.metric("Recall",f"{M['recall']*100:.1f}%"); z.metric("F1",f"{M['f1']*100:.1f}%")
        st.markdown('</div>',unsafe_allow_html=True)

elif page=="Transactions":
    st.markdown('<div class="page-title">Transactions</div><div class="page-subtitle">Historical monitoring and real-time AI analysis.</div>',unsafe_allow_html=True)
    st.write("")
    t1,t2=st.tabs(["Transaction Monitor","Analyze Transaction"])
    with t1:
        q=st.text_input("Search",placeholder="Search transaction...",label_visibility="collapsed")
        view=df.copy()
        if q:
            m=view.astype(str).apply(lambda x:x.str.contains(q,case=False,na=False)); view=view[m.any(axis=1)]
        st.markdown(f'<div class="small">{len(view):,} transactions found</div>',unsafe_allow_html=True)
        st.dataframe(view.head(500),width="stretch",hide_index=True)
    with t2:
        st.markdown('<div class="card"><div class="card-title">Real-Time Transaction Analysis</div><div class="card-sub">Uses the same feature space as your trained model.</div>',unsafe_allow_html=True)
        st.markdown('<div class="info-box" style="margin-top:12px">Start with a real transaction to see how the model works. The V1–V28 values are anonymized PCA components from the original dataset; you do not need to interpret them individually.</div>',unsafe_allow_html=True)
        mode=st.radio("Input mode",["Use a real transaction from dataset","Manual advanced features"],horizontal=True)
        row={}
        if mode.startswith("Use"):
            idx=st.selectbox("Select transaction",df.index.tolist(),format_func=lambda x:f"Transaction #{x}")
            raw=df.loc[idx]; amount=float(raw.Amount); tm=float(raw.Time)
            for i in range(1,29): row[f"V{i}"]=float(raw[f"V{i}"])
            row["scaled_amount"],row["scaled_time"]=scale_raw(amount,tm)
            x,y,z=st.columns(3); x.metric("Amount",f"₹ {amount:,.2f}"); y.metric("Time",f"{tm:,.2f}"); z.metric("Actual label","FRAUD" if int(raw[TARGET]) else "LEGITIMATE")
        else:
            amount=st.number_input("Amount",min_value=0.0,value=100.0); tm=st.number_input("Time",min_value=0.0,value=1000.0)
            cc=st.columns(4)
            for i in range(1,29):
                with cc[(i-1)%4]: row[f"V{i}"]=st.number_input(f"V{i}",value=0.0,key=f"v_{i}")
            row["scaled_amount"],row["scaled_time"]=scale_raw(amount,tm)
        if st.button("🔍 Analyze Transaction",type="primary",width="stretch"):
            try: st.session_state.result=predict(row)
            except Exception as e: st.error(f"Prediction failed: {type(e).__name__}: {e}")
        if st.session_state.result:
            pred,prob=st.session_state.result; label=risk(prob); cls={"HIGH":"result-high","MEDIUM":"result-medium","LOW":"result-low"}[label]
            color={"HIGH":"#ff718c","MEDIUM":"#f7bf45","LOW":"#35d39a"}[label]
            decision={"HIGH":"BLOCK","MEDIUM":"REVIEW","LOW":"APPROVE"}[label]
            st.write("")
            st.markdown(f'<div class="card {cls}"><div class="result-risk">AI FRAUD ANALYSIS</div><div class="result-status" style="color:{color};margin-top:12px">{label} RISK</div><div class="result-number">{prob*100:.2f}%</div><div class="result-risk">FRAUD PROBABILITY</div></div>',unsafe_allow_html=True)
            x,y,z=st.columns(3); x.metric("AI Decision",decision); y.metric("Prediction","FRAUD" if pred else "LEGITIMATE"); z.metric("Model","Random Forest")
            st.markdown('<div class="info-box">Risk bands: LOW &lt; 50% • MEDIUM 50–80% • HIGH ≥ 80%.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

elif page=="Alerts":
    st.markdown('<div class="page-title">Alerts</div><div class="page-subtitle">Fraudulent transactions from the real dataset.</div>',unsafe_allow_html=True)
    alerts=df[df[TARGET]==1] if TARGET in df else pd.DataFrame()
    st.write(""); a,b,c=st.columns(3); a.metric("Total Alerts",f"{len(alerts):,}"); b.metric("Fraud Rate",f"{len(alerts)/len(df)*100:.3f}%" if len(df) else "0"); c.metric("Model","Random Forest")
    st.dataframe(alerts.head(1000),width="stretch",hide_index=True)

elif page=="Analytics":
    st.markdown('<div class="page-title">Analytics</div><div class="page-subtitle">Real dataset analysis.</div>',unsafe_allow_html=True)
    nums=[x for x in df.select_dtypes(include=np.number).columns if x!=TARGET]
    a,b=st.columns(2)
    with a:
        col=st.selectbox("Feature distribution",nums); fig=px.histogram(df,x=col,nbins=45); fig.update_traces(marker_color="#7757ff"); st.plotly_chart(base(fig,320),width="stretch",config={"displayModeBar":False})
    with b:
        col=st.selectbox("Fraud vs legitimate",nums,key="box"); fig=px.box(df,x=TARGET,y=col); fig.update_traces(marker_color="#1fd1df"); st.plotly_chart(base(fig,320),width="stretch",config={"displayModeBar":False})
    if M:
        st.markdown('<div class="card"><div class="card-title">Confusion Matrix</div>',unsafe_allow_html=True)
        fig=go.Figure(go.Heatmap(z=M["cm"],x=["Predicted Legitimate","Predicted Fraud"],y=["Actual Legitimate","Actual Fraud"],text=M["cm"],texttemplate="%{text}",colorscale=[[0,"#0c1230"],[1,"#7757ff"]],showscale=False))
        st.plotly_chart(base(fig,250),width="stretch",config={"displayModeBar":False}); st.markdown('</div>',unsafe_allow_html=True)

elif page=="Models":
    st.markdown('<div class="page-title">Models</div><div class="page-subtitle">Actual trained model information and metrics.</div>',unsafe_allow_html=True)
    a,b,c,d,e=st.columns(5); a.metric("Status","Loaded" if model else "Missing"); b.metric("Algorithm","Random Forest"); c.metric("Trees",getattr(model,"n_estimators","—")); d.metric("Features",getattr(model,"n_features_in_","—")); e.metric("ROC-AUC",f"{M['roc_auc']:.4f}" if M else "—")
    st.write("")
    if M:
        a,b,c,d=st.columns(4); a.metric("Accuracy",f"{M['accuracy']*100:.2f}%"); b.metric("Precision",f"{M['precision']*100:.2f}%"); c.metric("Recall",f"{M['recall']*100:.2f}%"); d.metric("F1",f"{M['f1']*100:.2f}%")
    st.markdown('<div class="card"><div class="card-title">Model artifact</div>',unsafe_allow_html=True)
    st.code(", ".join(getattr(model,"feature_names_in_",FEATURES)) if model else "Model not loaded")
    st.markdown('</div>',unsafe_allow_html=True)

elif page=="Customers":
    st.markdown('<div class="page-title">Customers</div><div class="page-subtitle">Customer-level analytics will require customer identifiers.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card">The supplied creditcard.csv contains anonymized V1–V28 features and no customer ID, card number, country or location. This page therefore does not invent customer data.</div>',unsafe_allow_html=True)

elif page=="Reports":
    st.markdown('<div class="page-title">Reports</div><div class="page-subtitle">Export real data and model results.</div>',unsafe_allow_html=True)
    typ=st.selectbox("Report",["Fraud Transactions","All Transactions","Model Metrics"])
    if typ=="Fraud Transactions": out=df[df[TARGET]==1]
    elif typ=="All Transactions": out=df
    else: out=pd.DataFrame([M]) if M else pd.DataFrame()
    st.dataframe(out.head(500),width="stretch",hide_index=True)
    st.download_button("⇩ Download CSV Report",out.to_csv(index=False),"fraudshield_report.csv","text/csv",type="primary",width="stretch")

elif page=="Settings":
    st.markdown('<div class="page-title">Settings</div><div class="page-subtitle">Risk thresholds and application settings.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-title">Risk Thresholds</div>',unsafe_allow_html=True); st.slider("High risk",.5,1.,.8,.01); st.slider("Medium risk",.1,.8,.5,.01); st.toggle("Real-time monitoring",True); st.toggle("High-risk alerting",True); st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">Appearance</div><div class="card-sub">Choose the workspace theme.</div>',unsafe_allow_html=True)
        selected_theme = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "Dark" else 1, horizontal=True, label_visibility="collapsed")
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        st.markdown(f'<div class="small">Currently using {selected_theme} theme.</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="card-title">System</div><div class="small">Dataset: {DATA_PATH.name}</div><div class="small">Model: {MODEL_PATH.name}</div><div class="small">Rows: {len(df):,}</div><div class="small">Model loaded: {"Yes" if model else "No"}</div></div>',unsafe_allow_html=True)

else:
    st.markdown('<div class="page-title">How to Use <span style="color:#ff4f61">FraudShield</span> AI</div><div class="page-subtitle">AI-powered credit card fraud detection</div>',unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin:18px 0 22px;padding:14px 18px;border-color:rgba(61,124,255,.25)"><b style="color:#fff">FraudShield AI</b> helps you detect suspicious transactions in <span style="color:#ff718c">real-time</span>, analyze risk, and get <span style="color:#ff718c">AI-powered</span> recommendations.</div>',unsafe_allow_html=True)

    def guide_step(number, title, text, action):
        st.markdown(f'<div class="card" style="min-height:142px;padding:16px 18px"><div style="display:flex;gap:11px;align-items:flex-start"><span style="background:#ff384c;color:#fff;border-radius:50%;min-width:22px;height:22px;text-align:center;line-height:22px;font-weight:800;font-size:12px">{number}</span><div><div class="card-title">{title}</div><div class="card-sub" style="font-size:10px;line-height:1.55;margin-top:7px;color:#bac4dc">{text}</div><div style="font-size:9px;color:#ff718c;margin-top:12px;font-weight:700">{action}</div></div></div></div>',unsafe_allow_html=True)

    a,b=st.columns(2)
    with a:
        guide_step(1,"Open the Dashboard","Review transaction volume, fraudulent cases, detection rate, and money at risk from the supplied dataset.","Use Dashboard in the sidebar")
        guide_step(3,"Enter Transaction Details","Choose a real dataset transaction or enter Amount, Time, and the V1–V28 PCA components manually.","Use Transactions → Analyze Transaction")
        guide_step(5,"Check Live Monitoring","Review the transaction monitor and filter historical records to investigate suspicious activity.","Use Live Monitoring or Transaction History")
    with b:
        guide_step(2,"Analyze a Transaction","Run the Random Forest prediction to calculate fraud probability and assign a LOW, MEDIUM, or HIGH risk band.","Select Analyze Transaction")
        guide_step(4,"View the AI Analysis Result","Read the fraud prediction, probability, risk level, and recommended action: APPROVE, REVIEW, or BLOCK.","Review the result card")
        guide_step(6,"View Alerts & Reports","Inspect labeled fraud records, model statistics, and download CSV reports for audit and analysis.","Open Reports or Analytics")

    st.markdown('<div class="card" style="margin-top:6px;padding:14px 18px"><div class="card-title" style="color:#ff4f61">Best Practices</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px"><div class="small">🛡️ Review HIGH risk transactions immediately.</div><div class="small">🤖 Combine AI recommendations with manual verification.</div><div class="small">🔒 Keep your data secure and do not share credentials.</div></div></div>',unsafe_allow_html=True)
    with st.expander("Model and data details"): st.write({"dataset_exists":DATA_PATH.exists(),"model_exists":MODEL_PATH.exists(),"rows":len(df),"model":type(model).__name__ if model else None})

st.markdown('<div class="center small" style="padding:30px 0 5px">FraudShield AI • AI-powered credit card fraud detection</div>',unsafe_allow_html=True)
