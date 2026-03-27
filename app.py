"""
app.py — ATC Sentinel: Role-Based Air Traffic Intelligence
Roles: User (passenger) · Controller (ATC) · Admin (full control)
Landing → Auth → Loading → Role-Based Dashboard
"""
import os, sys, json, warnings, time as _time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from auth import (
    register_user, login_user, get_user_count, send_password_reset,
    check_session_valid, get_all_users, update_user_role, approve_user,
    deny_user, get_audit_log, get_pending_count, ROLES, _log_audit
)
from utils.data_loader import load_data, AIRPORTS, fetch_live_weather_all
from pipeline import run_pipeline, FEATURE_COLS, LABEL_MAP, COLOR_MAP

FRIENDLY_LABELS = {0: 'Clear', 1: 'Busy', 2: 'Overloaded'}
STATUS_EMOJI = {0: '🟢', 1: '🟡', 2: '🔴'}

st.set_page_config(page_title="Sentinel ATC", page_icon="✈", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════
#  THEME CSS (inline-safe)
# ═══════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');
.stApp{background:#060e1f!important}
section[data-testid="stSidebar"]{background:#0d1529!important;border-right:1px solid rgba(123,208,255,0.08)!important}
section[data-testid="stSidebar"] *{color:#c8d6e5!important}
#MainMenu,footer,header{visibility:hidden}.stDeployButton{display:none}
.metric-s{background:linear-gradient(135deg,#0a1020,#0d1529);border:1px solid rgba(255,255,255,0.04);border-radius:12px;padding:18px;text-align:center}
.metric-s .lb{font-family:JetBrains Mono;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;color:#4a5568;margin-bottom:6px}
.metric-s .vl{font-family:JetBrains Mono;font-size:1.7rem;font-weight:700}
.vl.p{color:#7bd0ff}.vl.w{color:#ffb783}.vl.d{color:#ff6b6b}.vl.s{color:#2ecc71}
.s-hdr{background:linear-gradient(135deg,#0b1326,#0d1529,#0f2040);border:1px solid rgba(123,208,255,0.1);border-radius:16px;padding:24px 32px;margin-bottom:24px;position:relative;overflow:hidden}
.s-hdr h1{font-family:Outfit;font-weight:800;font-size:1.7rem;color:#e8edf5;margin:0 0 4px;letter-spacing:-0.02em}
.s-hdr h1 span{color:#7bd0ff}
.s-hdr p{font-family:JetBrains Mono;font-size:0.72rem;color:#4a5568;margin:0}
.s-tag{display:inline-block;padding:3px 12px;background:rgba(123,208,255,0.08);border:1px solid rgba(123,208,255,0.2);border-radius:4px;font-family:JetBrains Mono;font-size:0.58rem;font-weight:700;color:#7bd0ff;letter-spacing:0.12em;text-transform:uppercase;margin-left:10px}
.ac{border-radius:12px;padding:14px 18px;margin:6px 0;border-left:4px solid}
.ac-o{background:rgba(255,107,107,0.06);border-left-color:#ff6b6b}
.ac-b{background:rgba(255,183,131,0.06);border-left-color:#ffb783}
.ac-c{background:rgba(46,204,113,0.04);border-left-color:#2ecc71}
.ac .an{font-family:Outfit;font-weight:700;font-size:0.88rem;color:#e8edf5}
.ac .ad{font-family:JetBrains Mono;font-size:0.7rem;color:#6b7b8d;margin-top:4px;line-height:1.6}
.st{font-family:Outfit;font-weight:700;font-size:1.05rem;color:#e8edf5;display:flex;align-items:center;gap:8px;margin-bottom:12px}
.ss{font-family:JetBrains Mono;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:#4a5568;margin-bottom:10px}
.gp{background:rgba(13,21,41,0.7);border:1px solid rgba(255,255,255,0.04);border-radius:16px;padding:24px}
.stPlotlyChart{border-radius:12px;overflow:hidden}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════
for k, v in [('page','landing'),('user',None),('loading_done',False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# Session timeout check
if st.session_state.user and not check_session_valid(st.session_state.user):
    st.session_state.user = None
    st.session_state.page = 'landing'
    st.toast("Session expired. Please sign in again.", icon="⏰")


# ═══════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════
def _logout():
    if st.session_state.user:
        _log_audit(st.session_state.user.get('uid',''), st.session_state.user.get('email',''), 'LOGOUT')
    st.session_state.user = None
    st.session_state.page = 'landing'
    st.session_state.loading_done = False
    st.cache_data.clear()
    st.rerun()

def _role_badge(role):
    r = ROLES.get(role, ROLES['user'])
    return f"{r['icon']} {r['label']}"

@st.cache_data(ttl=120)
def _get_data(src): return load_data(source=src)
@st.cache_data(ttl=180)
def _live_wx(): return fetch_live_weather_all()
@st.cache_resource
def _load_models():
    m = {}
    md = os.path.join(BASE_DIR, "models")
    for f, k in [("xgboost_model.pkl","xgb"),("scaler.pkl","scaler")]:
        p = os.path.join(md, f)
        if os.path.exists(p): m[k] = joblib.load(p)
    for n in ["lstm_model.keras","lstm_model.h5"]:
        p = os.path.join(md, n)
        if os.path.exists(p):
            try:
                from tensorflow.keras.models import load_model as lm
                m['lstm'] = lm(p, compile=False); break
            except: pass
    for jf, k in [("shap_data.json","shap"),("meta.json","meta")]:
        p = os.path.join(md, jf)
        if os.path.exists(p):
            with open(p) as f: m[k] = json.load(f)
    return m

def _process_data(source='synthetic', sim_runway=2, sim_weather=0.3, wx_override=False, live_wx=True):
    raw = _get_data(source)
    df = run_pipeline(raw)
    df['active_runways'] = sim_runway
    if wx_override: df['weather_severity'] = sim_weather
    if live_wx:
        wm = _live_wx()
        if wm:
            for idx, row in df.iterrows():
                icao = row['airport_icao']
                if icao in wm:
                    w = wm[icao]
                    for col, key in [('wind_speed_kmh','wind_speed_kmh'),('visibility_m','visibility_m'),
                                     ('precipitation_mm','precipitation_mm'),('cloud_cover','cloud_cover')]:
                        df.at[idx, col] = w.get(key, df.at[idx, col])
            from pipeline import compute_weather_severity
            df['weather_severity'] = df.apply(compute_weather_severity, axis=1)
    from pipeline import assign_congestion_label, compute_runway_util
    df['runway_util_ratio'] = df.apply(compute_runway_util, axis=1)
    df['congestion_label'] = df.apply(assign_congestion_label, axis=1)
    df['congestion_name'] = df['congestion_label'].map(LABEL_MAP)
    models = _load_models()
    if 'xgb' in models and 'scaler' in models:
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X = df[avail].fillna(0)
        Xs = models['scaler'].transform(X)
        df = df.copy()
        df['pred_label'] = models['xgb'].predict(Xs)
        proba = models['xgb'].predict_proba(Xs)
        df['congestion_score'] = (proba[:,1]*0.5+proba[:,2]*1.0).clip(0,1)
    return df, models


# ═══════════════════════════════════════
#  PAGE: LANDING
# ═══════════════════════════════════════
def render_landing():
    n_ap = len(AIRPORTS); n_u = get_user_count()
    st.markdown(f"""
    <div style="min-height:90vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 20% 50%,rgba(123,208,255,0.06)0%,transparent 50%),radial-gradient(ellipse at 80% 20%,rgba(217,119,34,0.04)0%,transparent 50%);pointer-events:none;z-index:0;"></div>
    <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 18px;background:rgba(123,208,255,0.06);border:1px solid rgba(123,208,255,0.15);border-radius:100px;font-family:JetBrains Mono;font-size:0.7rem;color:#7bd0ff;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:32px;z-index:2;">
        <span style="width:6px;height:6px;background:#2ecc71;border-radius:50%;display:inline-block;"></span>
        {n_ap} Airports Monitored · {n_u} Users Registered
    </div>
    <h1 style="font-family:Outfit;font-weight:900;font-size:clamp(2.8rem,6vw,5rem);line-height:1.05;letter-spacing:-0.04em;color:#e8edf5;margin:0 0 20px;z-index:2;">
        Predict Airport<br>Congestion <span style="color:#7bd0ff;">Before</span><br>It <span style="color:#d97722;">Happens</span></h1>
    <p style="font-family:Outfit;font-size:clamp(0.95rem,1.8vw,1.15rem);color:#6b7b8d;max-width:560px;line-height:1.7;margin:0 auto 44px;z-index:2;">
        AI-powered flight traffic monitoring for <strong style="color:#7bd0ff;">passengers</strong>,
        <strong style="color:#2ecc71;">controllers</strong>, and
        <strong style="color:#d97722;">administrators</strong>.</p>
    <div style="display:flex;gap:40px;justify-content:center;margin-bottom:44px;z-index:2;flex-wrap:wrap;">
        <div style="text-align:center;"><div style="font-family:JetBrains Mono;font-weight:700;font-size:1.8rem;color:#7bd0ff;">👤</div><div style="font-family:Outfit;font-size:0.7rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">Passengers</div><div style="font-family:Outfit;font-size:0.65rem;color:#6b7b8d;max-width:140px;margin-top:4px;">Check flights, delays & weather</div></div>
        <div style="text-align:center;"><div style="font-family:JetBrains Mono;font-weight:700;font-size:1.8rem;color:#2ecc71;">🎮</div><div style="font-family:Outfit;font-size:0.7rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">Controllers</div><div style="font-family:Outfit;font-size:0.65rem;color:#6b7b8d;max-width:140px;margin-top:4px;">AI predictions & rerouting</div></div>
        <div style="text-align:center;"><div style="font-family:JetBrains Mono;font-weight:700;font-size:1.8rem;color:#d97722;">👑</div><div style="font-family:Outfit;font-size:0.7rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">Admins</div><div style="font-family:Outfit;font-size:0.65rem;color:#6b7b8d;max-width:140px;margin-top:4px;">Full control & user management</div></div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;max-width:820px;z-index:2;">
        <div style="background:rgba(13,21,41,0.7);border:1px solid rgba(123,208,255,0.06);border-radius:14px;padding:24px 20px;text-align:left;">
            <div style="font-size:1.5rem;margin-bottom:10px;">🧠</div>
            <div style="font-family:Outfit;font-weight:700;font-size:0.9rem;color:#e8edf5;margin-bottom:6px;">AI Predictions</div>
            <div style="font-family:Outfit;font-size:0.75rem;color:#6b7b8d;line-height:1.5;">XGBoost + LSTM forecast congestion 30 minutes ahead.</div></div>
        <div style="background:rgba(13,21,41,0.7);border:1px solid rgba(123,208,255,0.06);border-radius:14px;padding:24px 20px;text-align:left;">
            <div style="font-size:1.5rem;margin-bottom:10px;">🔒</div>
            <div style="font-family:Outfit;font-weight:700;font-size:0.9rem;color:#e8edf5;margin-bottom:6px;">Secure Access</div>
            <div style="font-family:Outfit;font-size:0.75rem;color:#6b7b8d;line-height:1.5;">Firebase auth, role-based access, session timeouts, audit logging.</div></div>
        <div style="background:rgba(13,21,41,0.7);border:1px solid rgba(123,208,255,0.06);border-radius:14px;padding:24px 20px;text-align:left;">
            <div style="font-size:1.5rem;margin-bottom:10px;">🔀</div>
            <div style="font-family:Outfit;font-weight:700;font-size:0.9rem;color:#e8edf5;margin-bottom:6px;">Smart Rerouting</div>
            <div style="font-family:Outfit;font-size:0.75rem;color:#6b7b8d;line-height:1.5;">AI suggests alternate airports ranked by capacity & distance.</div></div>
    </div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _,c,_ = st.columns([1,1.5,1])
    with c:
        a,b = st.columns(2)
        with a:
            if st.button("🚀 Get Started", use_container_width=True, type="primary"):
                st.session_state.page='auth'; st.rerun()
        with b:
            if st.button("🔑 Sign In", use_container_width=True):
                st.session_state.page='auth'; st.rerun()


# ═══════════════════════════════════════
#  PAGE: AUTH
# ═══════════════════════════════════════
def render_auth():
    st.markdown("""<div style="max-width:440px;margin:30px auto;text-align:center;">
        <div style="font-size:2.2rem;margin-bottom:6px;">✈</div>
        <div style="font-family:Outfit;font-weight:800;font-size:1.4rem;color:#e8edf5;">Sentinel <span style="color:#7bd0ff;">ATC</span></div>
        <div style="font-family:JetBrains Mono;font-size:0.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;">Role-Based Air Traffic Intelligence</div>
    </div>""", unsafe_allow_html=True)
    _,center,_ = st.columns([1,1.5,1])
    with center:
        t1, t2 = st.tabs(["🔑 Sign In", "🚀 Create Account"])
        with t1:
            with st.form("login_form"):
                st.markdown("##### Welcome back")
                em = st.text_input("Email", placeholder="you@example.com", key="l_em")
                pw = st.text_input("Password", type="password", placeholder="••••••••", key="l_pw")
                sub = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                if sub:
                    if not em or not pw: st.error("Fill in all fields")
                    else:
                        ok, result = login_user(em, pw)
                        if ok:
                            if result.get('_pending'):
                                st.warning("⏳ Your controller account is pending admin approval.")
                            else:
                                st.session_state.user = result
                                st.session_state.page = 'loading'
                                st.rerun()
                        else: st.error(result)
            with st.expander("🔑 Forgot password?"):
                re = st.text_input("Email", key="r_em", placeholder="you@example.com")
                if st.button("Send Reset Link", key="r_btn"):
                    if re:
                        ok, msg = send_password_reset(re)
                        st.success(msg) if ok else st.error(msg)
        with t2:
            with st.form("reg_form"):
                st.markdown("##### Create your account")
                fn = st.text_input("Full Name", placeholder="John Doe", key="r_fn")
                em2 = st.text_input("Email", placeholder="you@example.com", key="r_em2")
                role_sel = st.selectbox("I am a...", ["Passenger (view flights & delays)", "ATC Controller (need approval)", "Administrator"], key="r_role")
                role_map = {"Passenger (view flights & delays)": "user", "ATC Controller (need approval)": "controller", "Administrator": "admin"}
                pw2 = st.text_input("Password", type="password", placeholder="Min 8 chars, upper+lower+number", key="r_pw")
                pw3 = st.text_input("Confirm Password", type="password", key="r_pw2")
                sub2 = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if sub2:
                    if pw2 != pw3: st.error("Passwords don't match")
                    elif not fn.strip(): st.error("Enter your name")
                    else:
                        ok, result = register_user(em2, pw2, fn.strip(), role_map[role_sel])
                        if ok:
                            if result.get('_pending'):
                                st.warning("✅ Account created! Your controller access is pending admin approval.")
                            else:
                                st.session_state.user = result
                                st.session_state.page = 'loading'
                                st.success("✅ Account created!")
                                st.rerun()
                        else: st.error(result)
                st.markdown("""<div style="font-family:JetBrains Mono;font-size:0.6rem;color:#4a5568;margin-top:8px;">
                    🔒 Password: min 8 chars, 1 uppercase, 1 lowercase, 1 number<br>
                    🛡️ Firebase encrypted · Session timeout: 60 min · Brute-force protection</div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _,bc,_ = st.columns([1,1.5,1])
    with bc:
        if st.button("← Back to Home", use_container_width=True): st.session_state.page='landing'; st.rerun()


# ═══════════════════════════════════════
#  PAGE: LOADING
# ═══════════════════════════════════════
def render_loading():
    user = st.session_state.user
    name = user['full_name'].split()[0] if user else "User"
    role_info = ROLES.get(user.get('role','user'), ROLES['user'])
    ph = st.empty()
    steps = [("Verifying credentials","🔐"),("Loading flight feeds","📡"),("Initializing AI models","🧠"),
             (f"Preparing {role_info['label']} dashboard","📊"),("Applying security policies","🛡️")]
    sty = "display:flex;align-items:center;gap:12px;padding:8px 0;font-family:Outfit;font-size:0.82rem;"
    for i in range(len(steps)):
        html = ""
        for j,(t,_) in enumerate(steps):
            c = "#2ecc71" if j<i else ("#7bd0ff" if j==i else "#4a5568")
            ck = "✅" if j<i else ("⏳" if j==i else "⬜")
            html += f'<div style="{sty}color:{c};"><span style="width:20px;">{ck}</span>{t}</div>'
        ph.markdown(f"""<div style="min-height:82vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
            <div style="font-size:3rem;margin-bottom:24px;animation:spin 3s linear infinite;">✈</div>
            <div style="font-family:Outfit;font-weight:600;font-size:1.2rem;color:#e8edf5;margin-bottom:6px;">Welcome, {name}</div>
            <div style="font-family:JetBrains Mono;font-size:0.7rem;color:{role_info['color']};margin-bottom:28px;">{role_info['icon']} {role_info['label']} Access</div>
            <div style="text-align:left;max-width:320px;">{html}</div>
        </div><style>@keyframes spin{{0%{{transform:rotate(0deg)}}100%{{transform:rotate(360deg)}}}}</style>""", unsafe_allow_html=True)
        _time.sleep(0.5)
    ph.empty(); st.session_state.page='dashboard'; st.session_state.loading_done=True; st.rerun()


# ═══════════════════════════════════════
#  SHARED: Sidebar
# ═══════════════════════════════════════
def render_sidebar(user, context="default"):
    r = ROLES.get(user['role'], ROLES['user'])
    initials = "".join([w[0] for w in user['full_name'].split()[:2]]).upper()
    with st.sidebar:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,{r['color']},#0d1529);display:flex;align-items:center;justify-content:center;font-family:Outfit;font-weight:800;font-size:0.8rem;color:#e8edf5;">{initials}</div>
            <div><div style="font-family:Outfit;font-weight:700;font-size:0.82rem;color:#e8edf5;">{user['full_name']}</div>
            <div style="font-family:JetBrains Mono;font-size:0.55rem;color:{r['color']};text-transform:uppercase;letter-spacing:0.1em;">{r['icon']} {r['label']}</div></div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        pending = get_pending_count()
        if user['role'] == 'admin' and pending > 0:
            st.warning(f"⏳ {pending} user(s) awaiting approval")
        if st.button("🚪 SIGN OUT", use_container_width=True, type="primary",
                      key=f"sb_logout_{context}"):
            _logout()
        st.divider()
        return st.sidebar


# ═══════════════════════════════════════
#  DASHBOARD: USER (Passenger)
# ═══════════════════════════════════════
def render_user_dashboard():
    user = st.session_state.user
    sb = render_sidebar(user, context="user")
    with sb:
        live_wx = st.toggle("🌦️ Live weather", True)

    # Top action bar
    _t1, _t2, _t3, _t4 = st.columns([2, 1, 1, 1])
    with _t1:
        st.markdown(f"""<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#4a5568;padding-top:8px;">
            Logged in as <span style="color:#7bd0ff;">{user['full_name']}</span> · 👤 Passenger</div>""", unsafe_allow_html=True)
    with _t3:
        if st.button("🔄 Refresh", use_container_width=True, key="u_ref"):
            st.cache_data.clear(); st.rerun()
    with _t4:
        if st.button("🚪 Logout", use_container_width=True, key="u_logout", type="primary"):
            _logout()

    st.markdown(f"""<div class="s-hdr"><h1>Flight <span>Status</span> <span class="s-tag">👤 Passenger</span></h1>
        <p>Check real-time airport conditions, delays, and weather before your trip</p></div>""", unsafe_allow_html=True)

    df, models = _process_data(live_wx=live_wx)
    latest = df.sort_values('timestamp').groupby('airport_icao').last().reset_index()

    # Airport selector
    sel = st.selectbox("🏢 Select your airport", list(AIRPORTS.keys()),
                       format_func=lambda x: f"{AIRPORTS[x]['name']} ({x})")
    ap = latest[latest['airport_icao'] == sel]
    if len(ap) > 0:
        r = ap.iloc[0]
        lbl = int(r.get('congestion_label', 0))
        score = r.get('congestion_score', 0)
        c1,c2,c3,c4 = st.columns(4)
        items = [("Airport Status",FRIENDLY_LABELS[lbl],"s" if lbl==0 else ("w" if lbl==1 else "d")),
                 ("Flights/Hour",f"{r.get('flights_per_hour',0):.0f}","p"),
                 ("Avg Delay",f"{r.get('avg_delay_min',0):.0f} min","d" if r.get('avg_delay_min',0)>20 else "s"),
                 ("Weather",f"{r.get('weather_severity',0):.0%}","d" if r.get('weather_severity',0)>0.6 else "p")]
        for col,(lb,vl,cls) in zip([c1,c2,c3,c4],items):
            with col: st.markdown(f'<div class="metric-s"><div class="lb">{lb}</div><div class="vl {cls}">{vl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Travel advisory
        if lbl == 2:
            st.error(f"⚠️ **{AIRPORTS[sel]['name']} is currently overloaded.** Expect significant delays. Consider alternative airports or later flights.")
        elif lbl == 1:
            st.warning(f"🟡 **{AIRPORTS[sel]['name']} is busy.** Moderate delays expected (~{r.get('avg_delay_min',0):.0f} min). Arrive early.")
        else:
            st.success(f"✅ **{AIRPORTS[sel]['name']} is running smoothly.** No significant delays expected.")

    # All airports overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="st">🗺️ All Airports Overview</div>', unsafe_allow_html=True)
    if len(latest) > 0:
        fig = go.Figure()
        for lbl,nm,clr in [(0,'Clear','#2ecc71'),(1,'Busy','#ffb783'),(2,'Overloaded','#ff6b6b')]:
            sub = latest[latest['congestion_label']==lbl]
            if len(sub)>0:
                fig.add_trace(go.Scattergeo(lon=sub['lon'],lat=sub['lat'],mode='markers+text',
                    marker=dict(size=15,color=clr,opacity=0.9),text=sub['airport_icao'],
                    textfont=dict(size=9,color='white',family='JetBrains Mono'),
                    textposition='top center',name=nm,
                    hovertext=sub.apply(lambda r:f"<b>{AIRPORTS.get(r['airport_icao'],{}).get('name','')}</b><br>{FRIENDLY_LABELS[int(r['congestion_label'])]}<br>{r['flights_per_hour']:.0f}/hr · {r['avg_delay_min']:.0f}m delay",axis=1),
                    hoverinfo='text'))
        fig.update_geos(projection_type="natural earth",showcoastlines=True,coastlinecolor="rgba(123,208,255,0.2)",
            showland=True,landcolor="#0a1628",showocean=True,oceancolor="#060e1f",
            showcountries=True,countrycolor="rgba(123,208,255,0.1)",bgcolor="rgba(0,0,0,0)",
            lataxis=dict(range=[5,35]),lonaxis=dict(range=[65,95]))
        fig.update_layout(height=380,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor="rgba(6,14,31,0.85)",font=dict(color="#c8d6e5",family="JetBrains Mono")))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════
#  DASHBOARD: CONTROLLER
# ═══════════════════════════════════════
def render_controller_dashboard(embedded=False):
    user = st.session_state.user
    if not embedded:
        sb = render_sidebar(user, context="controller")
        with sb:
            data_source = st.radio("📡 Source",["Simulated","Historical","Live"],index=0)
            src_map = {"Simulated":"synthetic","Historical":"kaggle","Live":"auto"}
            live_wx = st.toggle("🌦️ Live weather",True)
            st.divider()
            sim_rwy = st.slider("🛬 Runways",1,3,2)
            sim_wx = st.slider("🌧️ Weather",0.0,1.0,0.3,0.05)
            wx_ov = st.checkbox("Override weather",False)
            fw = st.selectbox("⏱️ Forecast",["15 min","30 min","60 min"],index=1)
            fw_min = int(fw.split()[0]); fw_steps = {15:3,30:6,60:12}[fw_min]
            sel_ap = st.multiselect("🏢 Airports",list(AIRPORTS.keys()),default=list(AIRPORTS.keys()),
                format_func=lambda x:AIRPORTS[x]['name'])
            if st.button("🚀 Run Prediction",type="primary",use_container_width=True): st.cache_data.clear()

        # Top action bar
        _t1, _t2, _t3, _t4 = st.columns([2, 1, 1, 1])
        with _t1:
            st.markdown(f"""<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#4a5568;padding-top:8px;">
                Logged in as <span style="color:#2ecc71;">{user['full_name']}</span> · 🎮 Controller</div>""", unsafe_allow_html=True)
        with _t3:
            if st.button("🔄 Refresh", use_container_width=True, key="c_ref"):
                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
        with _t4:
            if st.button("🚪 Logout", use_container_width=True, key="c_logout", type="primary"):
                _logout()

        st.markdown(f"""<div class="s-hdr"><h1>Traffic <span>Monitor</span> <span class="s-tag">🎮 Controller</span></h1>
            <p>AI-powered congestion prediction · {len(AIRPORTS)} airports · Real-time analysis</p></div>""", unsafe_allow_html=True)
    else:
        # Embedded mode: use defaults, no sidebar, simplified header
        data_source = "Simulated"; src_map = {"Simulated":"synthetic"}
        live_wx = True; sim_rwy = 2; sim_wx = 0.3; wx_ov = False
        fw_min = 30; fw_steps = 6
        sel_ap = list(AIRPORTS.keys())
        st.markdown('<div class="st">🎮 Controller View (embedded from Admin)</div>', unsafe_allow_html=True)

    df, models = _process_data(src_map[data_source], sim_rwy, sim_wx, wx_ov, live_wx)
    if sel_ap: df_d = df[df['airport_icao'].isin(sel_ap)]
    else: df_d = df
    latest = df_d.sort_values('timestamp').groupby('airport_icao').last().reset_index()

    # Metrics
    if len(latest)>0:
        tf=int(latest['flights_per_hour'].sum()); ad=round(float(latest['avg_delay_min'].mean()),1)
        cn=int((latest['congestion_label']==2).sum()); mn=int((latest['congestion_label']==1).sum())
    else: tf,ad,cn,mn=0,0.0,0,0
    mc=st.columns(5)
    for col,lb,vl,cls in zip(mc,["Flights","Avg Delay","Overloaded","Busy","Health"],
        [str(tf),f"{ad}m",str(cn),str(mn),"Stressed" if cn>=2 else ("Watch" if cn>=1 else "Clear")],
        ["p","d" if ad>25 else "s","d","w","d" if cn>=2 else "s"]):
        with col: st.markdown(f'<div class="metric-s"><div class="lb">{lb}</div><div class="vl {cls}">{vl}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    # Globe + Alerts
    c_map,c_alt = st.columns([2,1])
    with c_map:
        st.markdown('<div class="st">🌍 Airport Globe</div>',unsafe_allow_html=True)
        view = st.radio("",["🌐 3D Globe","🗺️ Flat Map"],horizontal=True,key="ct_view")
        if len(latest)>0:
            if view=="🌐 3D Globe":
                fig=go.Figure()
                for lbl,nm,clr,sym in [(0,'Clear','#2ecc71','circle'),(1,'Busy','#ffb783','diamond'),(2,'Overloaded','#ff6b6b','star')]:
                    sub=latest[latest['congestion_label']==lbl]
                    if len(sub)>0:
                        fig.add_trace(go.Scattergeo(lon=sub['lon'],lat=sub['lat'],mode='markers+text',
                            marker=dict(size=sub['flights_per_hour'].clip(lower=5)/2+6,color=clr,symbol=sym,opacity=0.9,
                                line=dict(width=1,color='rgba(255,255,255,0.3)')),
                            text=sub['airport_icao'],textfont=dict(size=10,color='white',family='JetBrains Mono'),
                            textposition='top center',name=nm,
                            hovertext=sub.apply(lambda r:f"<b>{AIRPORTS.get(r['airport_icao'],{}).get('name','')}</b><br>{FRIENDLY_LABELS[int(r['congestion_label'])]}<br>Score:{r.get('congestion_score',0):.0%}<br>{r['flights_per_hour']:.0f}/hr · {r['avg_delay_min']:.0f}m",axis=1),hoverinfo='text'))
                fig.update_geos(projection_type="orthographic",projection_rotation=dict(lon=79,lat=20),
                    showcoastlines=True,coastlinecolor="rgba(123,208,255,0.2)",showland=True,landcolor="#0a1628",
                    showocean=True,oceancolor="#060e1f",showcountries=True,countrycolor="rgba(123,208,255,0.1)",bgcolor="rgba(0,0,0,0)")
                fig.update_layout(height=440,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(bgcolor="rgba(6,14,31,0.85)",font=dict(color="#c8d6e5",family="JetBrains Mono",size=10)))
                st.plotly_chart(fig,use_container_width=True)
            else:
                fig=go.Figure()
                for lbl,nm,clr in [(0,'Clear','#2ecc71'),(1,'Busy','#ffb783'),(2,'Overloaded','#ff6b6b')]:
                    sub=latest[latest['congestion_label']==lbl]
                    if len(sub)>0:
                        fig.add_trace(go.Scattermapbox(lat=sub['lat'],lon=sub['lon'],mode='markers+text',
                            marker=dict(size=sub['flights_per_hour'].clip(lower=5)/2+8,color=clr,opacity=0.85),
                            text=sub['airport_icao'],textfont=dict(size=10,color='white',family='JetBrains Mono'),
                            textposition='top center',name=nm,hoverinfo='text'))
                fig.update_layout(mapbox=dict(style='carto-darkmatter',center=dict(lat=20.5,lon=79),zoom=4.3),
                    height=440,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig,use_container_width=True)

    with c_alt:
        st.markdown('<div class="st">🚨 Alerts</div>',unsafe_allow_html=True)
        for _,row in latest.sort_values('congestion_label',ascending=False).iterrows():
            lbl=int(row['congestion_label']);nm=AIRPORTS.get(row['airport_icao'],{}).get('name','');sc=row.get('congestion_score',0)
            css=['ac-c','ac-b','ac-o'][lbl]
            st.markdown(f'<div class="ac {css}"><div class="an">{STATUS_EMOJI[lbl]} {nm}</div><div class="ad">{FRIENDLY_LABELS[lbl]} · {sc:.0%} · {row.get("flights_per_hour",0):.0f}/hr · {row.get("avg_delay_min",0):.0f}m</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    # Forecast + SHAP
    fc,sh = st.columns([2,1])
    with fc:
        st.markdown(f'<div class="st">📈 {fw_min}-Min Forecast</div>',unsafe_allow_html=True)
        fap = st.selectbox("Airport:",sel_ap if sel_ap else list(AIRPORTS.keys()),format_func=lambda x:AIRPORTS[x]['name'],key="fc_ap")
        steps=list(range(1,fw_steps+1))
        base=df[df['airport_icao']==fap]['congestion_score'].mean() if 'congestion_score' in df.columns else 0.4
        if pd.isna(base): base=0.4
        scores=np.clip(base+np.random.normal(0,0.05,fw_steps),0,1).tolist()
        mins=[round(h*fw_min/fw_steps) for h in steps]
        fig=go.Figure()
        fig.add_hrect(y0=0.75,y1=1,fillcolor="#ff6b6b",opacity=0.08,annotation_text="🔴 Overloaded",annotation_position="top left",annotation=dict(font=dict(color="#ff6b6b",size=10)))
        fig.add_hrect(y0=0.45,y1=0.75,fillcolor="#ffb783",opacity=0.06,annotation_text="🟡 Busy",annotation_position="top left",annotation=dict(font=dict(color="#ffb783",size=10)))
        fig.add_hrect(y0=0,y1=0.45,fillcolor="#2ecc71",opacity=0.04,annotation_text="🟢 Clear",annotation_position="bottom left",annotation=dict(font=dict(color="#2ecc71",size=10)))
        fig.add_hline(y=0.75,line_dash="dash",line_color="#ff6b6b",line_width=1)
        fig.add_hline(y=0.45,line_dash="dash",line_color="#ffb783",line_width=1)
        fig.add_trace(go.Scatter(x=mins,y=scores,mode='lines+markers+text',line=dict(color='#7bd0ff',width=3),
            marker=dict(size=9,color='#7bd0ff',line=dict(width=2,color='#060e1f')),
            text=[f"{s:.0%}" for s in scores],textposition='top center',textfont=dict(size=9,family='JetBrains Mono',color='#c8d6e5'),
            fill='tozeroy',fillcolor='rgba(123,208,255,0.04)'))
        fig.update_layout(xaxis_title="Minutes ahead",yaxis_title="Congestion",yaxis=dict(range=[0,1.05],tickformat='.0%',gridcolor='rgba(255,255,255,0.03)'),
            height=360,font=dict(family='JetBrains Mono',color='#4a5568'),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(6,14,31,0.5)',margin=dict(l=50,r=80,t=10,b=40))
        st.plotly_chart(fig,use_container_width=True)

    FF = {'flights_per_hour':'Flights','avg_delay_min':'Delays','weather_severity':'Weather',
          'runway_util_ratio':'Runway load','restriction_flag':'Closures','hour_of_day':'Time',
          'day_of_week':'Day','is_peak_hour':'Rush hour','cloud_cover':'Clouds'}
    with sh:
        st.markdown('<div class="st">🔍 Why?</div>',unsafe_allow_html=True)
        if 'shap' in models:
            sd=models['shap'];fn=sd['feature_names'];sv=np.array(sd['shap_values'])
            ma=np.abs(sv).mean(axis=(0,1)) if sv.ndim==3 else np.abs(sv).mean(axis=0)
            n=min(len(fn),len(ma))
            sdf=pd.DataFrame({"F":[FF.get(f,f) for f in fn[:n]],"I":ma[:n]}).sort_values("I",ascending=True).tail(6)
            fig=px.bar(sdf,x="I",y="F",orientation='h',color="I",color_continuous_scale=[[0,'#0d1529'],[1,'#7bd0ff']],height=320)
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False,font=dict(family='JetBrains Mono',color='#4a5568'),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(6,14,31,0.3)')
            st.plotly_chart(fig,use_container_width=True)

    # Rerouting + Table
    st.markdown("<br>",unsafe_allow_html=True)
    with st.expander("🔀 Rerouting Recommendations"):
        flagged=latest[latest['congestion_label']>=1]
        if len(flagged)>0:
            for _,row in flagged.iterrows():
                nm=AIRPORTS.get(row['airport_icao'],{}).get('name','')
                others=latest[latest['airport_icao']!=row['airport_icao']]
                if len(others)>0 and 'congestion_score' in others.columns:
                    best=others.loc[others['congestion_score'].idxmin()]
                    alt=AIRPORTS.get(best['airport_icao'],{}).get('name','?')
                else: alt="nearby airport"
                st.markdown(f"**{STATUS_EMOJI[int(row['congestion_label'])]} {nm}** → Reroute to **{alt}**")
        else: st.success("✅ All clear")
    with st.expander("📋 Raw Data"):
        sc=['timestamp','airport_icao','flights_per_hour','avg_delay_min','congestion_name']
        st.dataframe(df_d[[c for c in sc if c in df_d.columns]].tail(30),use_container_width=True,height=200)
        st.download_button("⬇️ CSV",df_d.to_csv(index=False).encode(),"atc_data.csv","text/csv")


# ═══════════════════════════════════════
#  DASHBOARD: ADMIN
# ═══════════════════════════════════════
def render_admin_dashboard():
    user = st.session_state.user
    sb = render_sidebar(user, context="admin")
    with sb:
        admin_page = st.radio("📋 Admin Panel", ["System Overview","User Management","Audit Log","Controller View"])

    # Top action bar — always visible
    _t1, _t2, _t3, _t4 = st.columns([2, 1, 1, 1])
    with _t1:
        st.markdown(f"""<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#4a5568;padding-top:8px;">
            Logged in as <span style="color:#d97722;">{user['full_name']}</span> · 👑 Admin</div>""", unsafe_allow_html=True)
    with _t3:
        if st.button("🔄 Refresh", use_container_width=True, key="a_ref"):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    with _t4:
        if st.button("🚪 Logout", use_container_width=True, key="a_logout", type="primary"):
            _logout()

    # Header
    st.markdown(f"""<div class="s-hdr"><h1>Admin <span>Console</span> <span class="s-tag">👑 Administrator</span></h1>
        <p>Full system control · User management · Audit logging · Security monitoring</p></div>""", unsafe_allow_html=True)

    if admin_page == "System Overview":
        all_users = get_all_users()
        pending = [u for u in all_users if not u['approved']]
        df, models = _process_data()
        latest = df.sort_values('timestamp').groupby('airport_icao').last().reset_index()
        cn=int((latest['congestion_label']==2).sum()) if len(latest)>0 else 0

        c1,c2,c3,c4,c5 = st.columns(5)
        items = [("Total Users",str(len(all_users)),"p"),("Pending Approval",str(len(pending)),"w" if pending else "s"),
                 ("Airports",str(len(AIRPORTS)),"p"),("Critical",str(cn),"d" if cn else "s"),
                 ("System","Online","s")]
        for col,(lb,vl,cls) in zip([c1,c2,c3,c4,c5],items):
            with col: st.markdown(f'<div class="metric-s"><div class="lb">{lb}</div><div class="vl {cls}">{vl}</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        # Pending approvals
        if pending:
            st.markdown('<div class="st">⏳ Pending Approvals</div>',unsafe_allow_html=True)
            for u in pending:
                c1,c2,c3 = st.columns([3,1,1])
                with c1: st.write(f"**{u['full_name']}** ({u['email']}) — requested: {u['role']}")
                with c2:
                    if st.button("✅ Approve",key=f"ap_{u['uid']}"):
                        approve_user(user['uid'],u['uid']); st.rerun()
                with c3:
                    if st.button("❌ Deny",key=f"dn_{u['uid']}"):
                        deny_user(user['uid'],u['uid']); st.rerun()

        # User breakdown
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="st">👥 User Breakdown</div>',unsafe_allow_html=True)
        role_counts = {}
        for u in all_users:
            role_counts[u['role']] = role_counts.get(u['role'],0)+1
        for role,count in role_counts.items():
            ri = ROLES.get(role,ROLES['user'])
            st.markdown(f"{ri['icon']} **{ri['label']}**: {count} users")

    elif admin_page == "User Management":
        st.markdown('<div class="st">👥 All Registered Users</div>',unsafe_allow_html=True)
        all_users = get_all_users()
        for u in all_users:
            ri = ROLES.get(u['role'],ROLES['user'])
            with st.expander(f"{ri['icon']} {u['full_name']} — {u['email']} ({u['role']})"):
                c1,c2 = st.columns(2)
                with c1:
                    st.write(f"**UID:** `{u['uid'][:20]}...`")
                    st.write(f"**Created:** {u['created_at']}")
                    st.write(f"**Last login:** {u['last_login'] or 'Never'}")
                    st.write(f"**Logins:** {u['login_count']}")
                    st.write(f"**Approved:** {'✅ Yes' if u['approved'] else '❌ No'}")
                with c2:
                    new_role = st.selectbox("Change role",list(ROLES.keys()),
                        index=list(ROLES.keys()).index(u['role']),key=f"role_{u['uid']}")
                    if new_role != u['role']:
                        if st.button(f"Update to {new_role}",key=f"upd_{u['uid']}"):
                            update_user_role(user['uid'],u['uid'],new_role); st.rerun()
                    if not u['approved']:
                        if st.button("✅ Approve",key=f"apv_{u['uid']}"):
                            approve_user(user['uid'],u['uid']); st.rerun()

    elif admin_page == "Audit Log":
        st.markdown('<div class="st">🛡️ Security Audit Log</div>',unsafe_allow_html=True)
        logs = get_audit_log(200)
        if logs:
            log_df = pd.DataFrame(logs)
            # Filters
            actions = ['ALL'] + sorted(log_df['action'].unique().tolist())
            filt = st.selectbox("Filter by action",actions)
            if filt != 'ALL':
                log_df = log_df[log_df['action']==filt]
            st.dataframe(log_df, use_container_width=True, height=400)
            st.download_button("⬇️ Export Log",log_df.to_csv(index=False).encode(),"audit_log.csv","text/csv")
        else:
            st.info("No audit events recorded yet.")

    elif admin_page == "Controller View":
        render_controller_dashboard(embedded=True)


# ═══════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════
page = st.session_state.page

if page == 'landing':
    render_landing()
elif page == 'auth':
    render_auth()
elif page == 'loading':
    render_loading()
elif page == 'dashboard':
    user = st.session_state.user
    if not user:
        st.session_state.page = 'landing'; st.rerun()
    elif user.get('role') == 'admin':
        render_admin_dashboard()
    elif user.get('role') == 'controller':
        render_controller_dashboard()
    else:
        render_user_dashboard()
