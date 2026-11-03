"""NYC Stay Classifier - Streamlit front end for the FastAPI room-type model.

Run:   streamlit run app.py
API:   uvicorn main:app --reload      (default http://127.0.0.1:8000)
"""
import os
import random
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from nyc_data import BOROUGH_CENTER, BOROUGH_HOODS, coords_for

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="NYC Stay Classifier", page_icon="🏙️", layout="wide")

MODEL_PATH = Path(__file__).parent / "Model_Pipeline.pkl"
DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")
COLUMNS = [
    "latitude", "longitude", "price", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",
]
CLASSES = ["Entire home/apt", "Private room", "Shared room"]
META = {
    "Entire home/apt": {"emoji": "🏠", "color": "#FFB454", "blurb": "The guest gets the whole place to themselves."},
    "Private room": {"emoji": "🛏️", "color": "#4FD1C5", "blurb": "Guests get their own room inside a shared home."},
    "Shared room": {"emoji": "🛋️", "color": "#FF7A8A", "blurb": "Guests sleep in a space shared with other people."},
}
PRESETS = {
    "Midtown studio": dict(borough="Manhattan", hood="Midtown", price=185, min_nights=2,
                           availability=220, n_reviews=120, rpm=2.1, host_count=1),
    "Brooklyn spare room": dict(borough="Brooklyn", hood="Bedford-Stuyvesant", price=70, min_nights=3,
                                availability=140, n_reviews=48, rpm=1.2, host_count=2),
    "Queens budget bed": dict(borough="Queens", hood="Jackson Heights", price=35, min_nights=1,
                              availability=330, n_reviews=12, rpm=0.6, host_count=6),
}

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');
:root{--ink:#0B1020;--panel:#131A33;--line:#252F57;--text:#E8ECFF;--muted:#8D97C2;--amber:#FFB454;--teal:#4FD1C5;--coral:#FF7A8A;}
.stApp{background:radial-gradient(1200px 600px at 80% -10%,#1b2450 0%,var(--ink) 60%);}
.stApp, .stMarkdown, label, input, button, p, li, textarea{font-family:'DM Sans',system-ui,sans-serif;}
h1,h2,h3,.hero h1,.result-label{font-family:'Bricolage Grotesque','DM Sans',sans-serif;letter-spacing:-0.02em;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;max-width:1200px;}
[data-testid="stSidebar"]{background:#0E1530;border-right:1px solid var(--line);}

/* ---------- hero + skyline (the one always-on animation) ---------- */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:22px;
  background:linear-gradient(180deg,#151d42 0%,#0d1330 100%);padding:34px 36px 0 36px;margin-bottom:22px;}
.hero h1{font-size:clamp(1.9rem,4vw,3.1rem);font-weight:800;line-height:1.05;margin:0 0 10px 0;color:var(--text);max-width:16em;}
.hero p{color:var(--muted);font-size:1.05rem;max-width:44em;margin:0 0 18px 0;}
.skyline{display:flex;justify-content:center;align-items:flex-end;gap:3px;height:140px;margin:0 -40px;}
.bld{flex:0 0 auto;display:flex;flex-wrap:wrap;align-content:flex-start;gap:5px;padding:7px 6px;
  background:linear-gradient(180deg,#1c2552,#121936);border-radius:3px 3px 0 0;box-sizing:border-box;}
.w{display:block;width:5px;height:7px;background:#232d5c;border-radius:1px;}
.w.lit{background:#7a6a4a;}
.w.on{background:var(--amber);box-shadow:0 0 6px rgba(255,180,84,.7);animation:twinkle 5s ease-in-out infinite;}
@keyframes twinkle{0%,100%{opacity:.12}50%{opacity:1}}

/* ---------- section cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;}
.section-title{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.15rem;margin:0 0 2px 0;color:var(--text);}
.section-sub{color:var(--muted);font-size:.9rem;margin:0 0 8px 0;}

/* ---------- buttons ---------- */
.stButton > button[kind="primary"]{background:linear-gradient(135deg,#FFB454,#FF8F5A);color:#1a1205;font-weight:700;
  border:0;border-radius:12px;padding:.7rem 1rem;transition:transform .15s ease,box-shadow .15s ease;}
.stButton > button[kind="primary"]:hover{transform:translateY(-2px);box-shadow:0 10px 28px rgba(255,180,84,.35);color:#1a1205;}
.stButton > button[kind="secondary"]{border-radius:999px;border:1px solid var(--line);background:var(--panel);transition:border-color .15s ease,transform .15s ease;}
.stButton > button[kind="secondary"]:hover{border-color:var(--amber);transform:translateY(-1px);}

/* ---------- result (replays because its container key changes per prediction) ---------- */
.result{display:flex;align-items:center;gap:18px;padding:20px 22px;border-radius:18px;
  border:1px solid color-mix(in srgb,var(--accent) 55%,transparent);
  background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 16%,#131A33),#131A33);
  animation:pop .55s cubic-bezier(.2,.9,.3,1.2) both;}
.result .emoji{font-size:3rem;line-height:1;animation:bob 2.6s ease-in-out .6s infinite;}
.result-label{font-size:1.7rem;font-weight:800;color:var(--accent);line-height:1.1;}
.result-sub{color:var(--muted);font-size:.95rem;margin-top:4px;}
.conf{margin-left:auto;text-align:right;font-family:'Bricolage Grotesque',sans-serif;font-size:2.2rem;font-weight:800;color:var(--text);}
.conf span{display:block;font-family:'DM Sans',sans-serif;font-size:.8rem;font-weight:500;color:var(--muted);}
@keyframes pop{from{opacity:0;transform:translateY(14px) scale(.97)}to{opacity:1;transform:none}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.bars{margin-top:16px;display:grid;gap:10px;}
.bar-row{display:grid;grid-template-columns:130px 1fr 52px;align-items:center;gap:12px;font-size:.92rem;color:var(--text);}
.bar-row .pct{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;}
.track{height:12px;background:#0d1330;border-radius:999px;overflow:hidden;border:1px solid var(--line);}
.fill{height:100%;width:var(--w);background:var(--c);border-radius:999px;animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d);}
@keyframes grow{from{width:0}to{width:var(--w)}}
.note{margin-top:12px;color:var(--muted);font-size:.88rem;}
.empty{border:1px dashed var(--line);border-radius:18px;padding:26px;color:var(--muted);text-align:center;}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.8rem;border:1px solid var(--line);}
.pill.ok{color:var(--teal);border-color:rgba(79,209,197,.5);}
.pill.bad{color:var(--coral);border-color:rgba(255,122,138,.5);}

@media (max-width:700px){.bar-row{grid-template-columns:100px 1fr 44px}.result{flex-wrap:wrap}.conf{margin-left:0;text-align:left}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def skyline_html() -> str:
    rnd = random.Random(7)
    buildings = []
    for _ in range(28):
        w, h = rnd.randint(34, 70), rnd.randint(40, 130)
        wins = []
        for _ in range(max(2, w // 11) * max(2, h // 12)):
            r = rnd.random()
            if r < 0.30:
                wins.append(f'<i class="w on" style="animation-delay:-{rnd.uniform(0, 8):.1f}s;'
                            f'animation-duration:{rnd.uniform(3, 8):.1f}s"></i>')
            elif r < 0.55:
                wins.append('<i class="w lit"></i>')
            else:
                wins.append('<i class="w"></i>')
        buildings.append(f'<div class="bld" style="width:{w}px;height:{h}px">{"".join(wins)}</div>')
    return f'<div class="skyline">{"".join(buildings)}</div>'


@st.cache_data(show_spinner=False)
def hero_html() -> str:
    return (
        '<div class="hero"><h1>What kind of stay is this New York listing?</h1>'
        "<p>Describe an Airbnb listing and the model predicts whether it is an entire home, "
        "a private room or a shared room.</p>" + skyline_html() + "</div>"
    )


st.markdown(hero_html(), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Model access (FastAPI first, local pickle as fallback)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_local_model():
    return joblib.load(MODEL_PATH)


def predict_via_api(payload: dict, base_url: str) -> dict:
    r = requests.post(f"{base_url.rstrip('/')}/predict", json=payload, timeout=10)
    if r.status_code == 422:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in r.json().get("detail", []))
        raise ValueError(f"The API rejected the input ({problems})")
    r.raise_for_status()
    data = r.json()
    return dict(zip(CLASSES, data["Probability"]))


def predict_locally(payload: dict) -> dict:
    model = load_local_model()
    proba = model.predict_proba(pd.DataFrame([payload], columns=COLUMNS))[0]
    return dict(zip(model.classes_, proba))


def predict(payload: dict) -> tuple[dict, str]:
    """Return ({class: probability}, source description)."""
    if st.session_state.mode == "FastAPI backend":
        try:
            return predict_via_api(payload, st.session_state.api_url), f"FastAPI at {st.session_state.api_url}"
        except requests.exceptions.RequestException:
            if not MODEL_PATH.exists():
                raise
            return predict_locally(payload), "local model (the API was unreachable)"
    return predict_locally(payload), "local model file"


# --------------------------------------------------------------------------- #
# Session state + callbacks
# --------------------------------------------------------------------------- #
DEFAULTS = dict(borough="Manhattan", hood="Midtown", price=150, min_nights=2, availability=180,
                n_reviews=25, rpm=1.0, host_count=1, mode="FastAPI backend", api_url=DEFAULT_API,
                history=[], run=0, last=None, sweep=None)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
if "lat" not in st.session_state:
    st.session_state.lat, st.session_state.lon = coords_for(st.session_state.borough, st.session_state.hood)


def sync_coords():
    st.session_state.lat, st.session_state.lon = coords_for(st.session_state.borough, st.session_state.hood)


def on_borough_change():
    st.session_state.hood = BOROUGH_HOODS[st.session_state.borough][0]
    sync_coords()


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v
    sync_coords()
    st.session_state.last = None
    st.session_state.sweep = None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Connection")
    st.radio("Predict with", ["FastAPI backend", "Local model file"], key="mode",
             help="The local option loads Model_Pipeline.pkl directly, with no server needed.")
    if st.session_state.mode == "FastAPI backend":
        st.text_input("API URL", key="api_url")
        if st.button("Test connection", width="stretch"):
            try:
                requests.get(st.session_state.api_url.rstrip("/") + "/", timeout=3).raise_for_status()
                st.markdown('<span class="pill ok">API is reachable</span>', unsafe_allow_html=True)
            except requests.exceptions.RequestException:
                hint = " Predictions will use the local model." if MODEL_PATH.exists() else ""
                st.markdown(f'<span class="pill bad">Cannot reach the API.{hint}</span>', unsafe_allow_html=True)
    elif not MODEL_PATH.exists():
        st.error("Model_Pipeline.pkl was not found next to app.py.")
    st.divider()
    st.markdown("### About the model")
    st.caption("Random forest trained on New York Airbnb listings. It uses 10 inputs, "
               "including 217 neighbourhoods across 5 boroughs, and predicts 3 room types.")

# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
st.markdown('<div class="section-title">Try an example</div>', unsafe_allow_html=True)
pcols = st.columns(len(PRESETS))
for col, name in zip(pcols, PRESETS):
    col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), width="stretch")

left, right = st.columns([1.1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Where is it?</div>'
                    '<div class="section-sub">Picking a neighbourhood moves the pin. You can still adjust the coordinates.</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.selectbox("Borough", list(BOROUGH_HOODS), key="borough", on_change=on_borough_change)
        c2.selectbox("Neighbourhood", BOROUGH_HOODS[st.session_state.borough], key="hood", on_change=sync_coords)
        c3, c4 = st.columns(2)
        c3.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.001, format="%.5f", key="lat")
        c4.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.001, format="%.5f", key="lon")

    with st.container(border=True):
        st.markdown('<div class="section-title">The listing</div>'
                    '<div class="section-sub">What a guest would see before booking.</div>', unsafe_allow_html=True)
        st.slider("Price per night ($)", 10, 1500, key="price", step=5)
        c5, c6 = st.columns(2)
        c5.slider("Minimum nights", 1, 365, key="min_nights")
        c6.slider("Days available this year", 0, 365, key="availability")

    with st.container(border=True):
        st.markdown('<div class="section-title">Host and reviews</div>'
                    '<div class="section-sub">Signals about how active the listing is.</div>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        c7.number_input("Total reviews", min_value=0, step=1, key="n_reviews")
        c8.number_input("Reviews per month", min_value=0.0, step=0.1, format="%.2f", key="rpm")
        c9.number_input("Host's listings", min_value=0, step=1, key="host_count")

    go = st.button("Predict room type", type="primary", width="stretch")

payload = dict(
    latitude=float(st.session_state.lat), longitude=float(st.session_state.lon),
    price=float(st.session_state.price), minimum_nights=int(st.session_state.min_nights),
    number_of_reviews=int(st.session_state.n_reviews), reviews_per_month=float(st.session_state.rpm),
    calculated_host_listings_count=int(st.session_state.host_count),
    availability_365=int(st.session_state.availability),
    neighbourhood_group=st.session_state.borough, neighbourhood=st.session_state.hood,
)

if go:
    try:
        with st.spinner("Reading the listing..."):
            probs, source = predict(payload)
    except (requests.exceptions.RequestException, ValueError) as exc:
        st.session_state.last = None
        st.error(f"Prediction failed. {exc}. Start the API with `uvicorn main:app --reload`, "
                 "or switch to the local model in the sidebar.")
    else:
        label = max(probs, key=probs.get)
        st.session_state.run += 1
        st.session_state.last = dict(probs=probs, label=label, source=source)
        st.session_state.sweep = None
        st.session_state.history.insert(0, {
            "Time": datetime.now().strftime("%H:%M:%S"), "Borough": payload["neighbourhood_group"],
            "Neighbourhood": payload["neighbourhood"], "Price": payload["price"],
            "Min nights": payload["minimum_nights"], "Available": payload["availability_365"],
            "Prediction": label, "Confidence": round(probs[label] * 100, 1),
        })
        st.toast(f"{META[label]['emoji']} {label}", icon="✅")

with right:
    last = st.session_state.last
    if last is None:
        st.markdown('<div class="empty"><div style="font-size:2rem">🌆</div>'
                    "Fill in the listing, then press <b>Predict room type</b>.<br>"
                    "Or start from one of the examples above.</div>", unsafe_allow_html=True)
    else:
        probs, label = last["probs"], last["label"]
        meta = META[label]
        top2 = sorted(probs, key=probs.get, reverse=True)[:2]
        # A new key per prediction creates a fresh DOM node, so the entrance animation replays.
        with st.container(key=f"result_{st.session_state.run}"):
            bars = "".join(
                f'<div class="bar-row"><span>{c}</span><div class="track"><div class="fill" '
                f'style="--w:{probs[c] * 100:.1f}%;--c:{META[c]["color"]};--d:{i * 0.12:.2f}s"></div></div>'
                f'<span class="pct">{probs[c] * 100:.1f}%</span></div>'
                for i, c in enumerate(CLASSES)
            )
            close_call = ""
            if probs[top2[0]] - probs[top2[1]] < 0.15:
                close_call = (f'<div class="note">Close call: the model is split between '
                              f"<b>{top2[0]}</b> and <b>{top2[1]}</b>.</div>")
            st.markdown(
                f'<div class="result" style="--accent:{meta["color"]}"><div class="emoji">{meta["emoji"]}</div>'
                f'<div><div class="result-label">{label}</div><div class="result-sub">{meta["blurb"]}</div></div>'
                f'<div class="conf">{probs[label] * 100:.0f}%<span>confidence</span></div></div>'
                f'<div class="bars">{bars}</div>{close_call}'
                f'<div class="note">Answered by {last["source"]}.</div>',
                unsafe_allow_html=True,
            )

    st.pydeck_chart(pdk.Deck(
        map_style="dark",
        initial_view_state=pdk.ViewState(latitude=payload["latitude"], longitude=payload["longitude"],
                                         zoom=11.5, pitch=45),
        layers=[
            pdk.Layer("ScatterplotLayer", data=[{"lat": payload["latitude"], "lon": payload["longitude"]}],
                      get_position="[lon, lat]", get_radius=700, get_fill_color=[255, 180, 84, 50]),
            pdk.Layer("ScatterplotLayer", data=[{"lat": payload["latitude"], "lon": payload["longitude"]}],
                      get_position="[lon, lat]", get_radius=160, get_fill_color=[255, 180, 84, 255]),
        ],
    ), height=300)

# --------------------------------------------------------------------------- #
# Explore tabs
# --------------------------------------------------------------------------- #
tab_sweep, tab_history = st.tabs(["How price changes the answer", "Prediction history"])

with tab_sweep:
    st.caption("Keeps everything else fixed and re-predicts across a range of nightly prices.")
    if st.button("Run price sweep"):
        prices = [20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 500, 700, 1000]
        rows = []
        try:
            with st.spinner("Testing 14 price points..."):
                for p in prices:
                    probs, _ = predict({**payload, "price": float(p)})
                    rows += [{"price": p, "room_type": c, "probability": v} for c, v in probs.items()]
            st.session_state.sweep = dict(df=pd.DataFrame(rows), at=payload["price"])
        except (requests.exceptions.RequestException, ValueError) as exc:
            st.error(f"Price sweep failed. {exc}")
    sweep = st.session_state.sweep
    if sweep:
        base = alt.Chart(sweep["df"]).encode(
            x=alt.X("price:Q", title="Price per night ($)", scale=alt.Scale(type="log")),
            y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("room_type:N", legend=alt.Legend(title=None, orient="bottom"),
                            scale=alt.Scale(domain=CLASSES, range=[META[c]["color"] for c in CLASSES])),
        )
        line = base.mark_line(point=True, strokeWidth=3, interpolate="monotone")
        marker = alt.Chart(pd.DataFrame({"price": [sweep["at"]]})).mark_rule(
            strokeDash=[5, 5], color="#E8ECFF").encode(x="price:Q")
        st.altair_chart((line + marker).properties(height=320))
        st.caption("The dashed line marks the price you entered.")

with tab_history:
    hist = st.session_state.history
    if not hist:
        st.info("Predictions you make appear here.")
    else:
        df_hist = pd.DataFrame(hist)
        st.dataframe(df_hist, width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download CSV", df_hist.to_csv(index=False).encode(), "predictions.csv",
                           "text/csv", width="stretch")
        if d2.button("Clear history", width="stretch"):
            st.session_state.history = []
            st.rerun()
