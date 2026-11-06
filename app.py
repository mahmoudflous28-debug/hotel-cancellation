import json, datetime as dt
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
import hotel_features  # noqa: F401  (needed so joblib can unpickle build_features)

st.set_page_config(page_title="Hotel Cancellation Risk", page_icon="🏨", layout="wide")
BASE = Path(__file__).parent

@st.cache_resource
def load():
    return (joblib.load(BASE / "model" / "hotel_cancel_pipeline.joblib"),
            json.loads((BASE / "model" / "meta.json").read_text()))
pipe, meta = load()
LOW, HIGH = meta["low"], meta["high"]
CH = meta["choices"]

POLICY = {
    "Low": ("🟢 Low risk", "**No deposit.** Keep the booking flexible — this guest is very likely to arrive."),
    "Medium": ("🟠 Medium risk", "**Partial deposit** (e.g. the first night) and send a reminder 7 days before arrival."),
    "High": ("🔴 High risk", "**Full / non-refundable deposit**, or count this room as a candidate for overbooking."),
}
def tier(p): return "Low" if p < LOW else ("Medium" if p < HIGH else "High")

def to_raw(f: dict) -> pd.DataFrame:
    d = f["arrival"]
    return pd.DataFrame([{
        "hotel": f["hotel"], "lead_time": f["lead_time"], "arrival_date_year": d.year,
        "arrival_date_month": d.strftime("%B"), "arrival_date_week_number": d.isocalendar()[1],
        "arrival_date_day_of_month": d.day, "stays_in_weekend_nights": f["weekend"], "stays_in_week_nights": f["week"],
        "adults": f["adults"], "children": f["children"], "babies": f["babies"], "meal": f["meal"],
        "country": f["country"], "market_segment": f["segment"], "distribution_channel": f["channel"],
        "is_repeated_guest": int(f["repeat"]), "previous_cancellations": f["prev_canc"],
        "previous_bookings_not_canceled": f["prev_ok"], "reserved_room_type": f["room"],
        "days_in_waiting_list": f["waitlist"], "customer_type": f["ctype"], "adr": f["adr"],
        "total_of_special_requests": f["requests"], "agent": 1.0 if f["agent"] else np.nan,
        "company": 1.0 if f["company"] else np.nan}])

def what_if(raw: pd.DataFrame, base_p: float):
    """Change one thing at a time and see how the risk moves (simple local explanation)."""
    tweaks = {
        "Shorter lead time (30 days)": lambda r: r.assign(lead_time=30),
        "Longer lead time (180 days)": lambda r: r.assign(lead_time=180),
        "Guest adds a special request": lambda r: r.assign(total_of_special_requests=max(1, int(r.total_of_special_requests.iloc[0]))),
        "No special requests": lambda r: r.assign(total_of_special_requests=0),
        "Booked directly": lambda r: r.assign(market_segment="Direct", distribution_channel="Direct"),
        "Booked via online travel agency": lambda r: r.assign(market_segment="Online TA", distribution_channel="TA/TO"),
        "Repeat guest": lambda r: r.assign(is_repeated_guest=1),
        "First-time guest": lambda r: r.assign(is_repeated_guest=0),
        "No previous cancellations": lambda r: r.assign(previous_cancellations=0),
    }
    out = []
    for name, fn in tweaks.items():
        p = float(pipe.predict_proba(fn(raw.copy()))[:, 1][0])
        if abs(p - base_p) >= 0.02: out.append((name, p - base_p))
    return sorted(out, key=lambda x: -abs(x[1]))[:4]

st.title("🏨 Hotel Booking Cancellation Risk")
st.caption("Predicts how likely a booking is to be cancelled and suggests a deposit policy. "
           "Built on ~87K real bookings from two Portuguese hotels. A decision-support tool — not a guarantee.")
tab1, tab2, tab3 = st.tabs(["Single booking", "Upload bookings (CSV)", "About the model"])

with tab1:
    with st.form("booking"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Stay")
            hotel = st.selectbox("Hotel", CH["hotel"])
            arrival = st.date_input("Arrival date", dt.date.today() + dt.timedelta(days=60))
            lead = st.number_input("Days between booking and arrival (lead time)", 0, 800, 60)
            week = st.number_input("Week nights", 0, 30, 2)
            weekend = st.number_input("Weekend nights", 0, 15, 1)
            meal = st.selectbox("Meal plan", CH["meal"], index=CH["meal"].index("BB") if "BB" in CH["meal"] else 0)
            room = st.selectbox("Reserved room type", CH["reserved_room_type"])
        with c2:
            st.subheader("Guest")
            adults = st.number_input("Adults", 0, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            babies = st.number_input("Babies", 0, 5, 0)
            country = st.selectbox("Country (ISO code)", sorted(set(meta["top_countries"] + ["Unknown"])),
                                   index=sorted(set(meta["top_countries"] + ["Unknown"])).index("PRT"))
            ctype = st.selectbox("Customer type", CH["customer_type"])
            repeat = st.checkbox("Repeat guest")
            prev_canc = st.number_input("Previous cancellations", 0, 30, 0)
            prev_ok = st.number_input("Previous completed stays", 0, 80, 0)
        with c3:
            st.subheader("Booking")
            segment = st.selectbox("Market segment", CH["market_segment"], index=CH["market_segment"].index("Online TA"))
            channel = st.selectbox("Distribution channel", CH["distribution_channel"],
                                   index=CH["distribution_channel"].index("TA/TO") if "TA/TO" in CH["distribution_channel"] else 0)
            adr = st.number_input("Average daily rate (ADR, €)", 0.0, 1000.0, float(round(meta["ranges"]["adr_median"])), step=5.0)
            requests = st.number_input("Special requests so far", 0, 5, 0)
            waitlist = st.number_input("Days on waiting list", 0, 400, 0)
            agent = st.checkbox("Booked through an agent", value=True)
            company = st.checkbox("Booked through a company")
        go = st.form_submit_button("Predict cancellation risk", type="primary", use_container_width=True)

    if go:
        errors = []
        if adults + children + babies < 1: errors.append("At least one guest is required.")
        if adults < 1 and children + babies > 0: errors.append("Children/babies need at least one adult.")
        if week + weekend < 1: errors.append("The stay must be at least one night.")
        if arrival < dt.date.today() - dt.timedelta(days=1): st.warning("Arrival date is in the past — result is still computed.")
        if errors:
            for e in errors: st.error(e)
        else:
            try:
                raw = to_raw(dict(hotel=hotel, arrival=arrival, lead_time=lead, weekend=weekend, week=week, adults=adults,
                    children=children, babies=babies, meal=meal, country=country, segment=segment, channel=channel,
                    repeat=repeat, prev_canc=prev_canc, prev_ok=prev_ok, room=room, waitlist=waitlist, ctype=ctype,
                    adr=adr, requests=requests, agent=agent, company=company))
                p = float(pipe.predict_proba(raw)[:, 1][0]); t = tier(p)
                label, advice = POLICY[t]
                a, b = st.columns([1, 2])
                a.metric("Cancellation probability", f"{p:.0%}"); a.progress(min(max(p, 0.0), 1.0)); a.markdown(f"### {label}")
                b.markdown("#### Recommended policy"); b.markdown(advice)
                b.caption(f"Revenue at stake for this stay: ≈ €{adr * (week + weekend):,.0f}  ·  Tiers: Low < {LOW:.0%}, Medium {LOW:.0%}–{HIGH:.0%}, High > {HIGH:.0%}")
                wi = what_if(raw, p)
                if wi:
                    st.markdown("#### What moves this booking's risk?")
                    for name, dlt in wi: st.write(f"{'🔺' if dlt > 0 else '🔻'} **{name}** → {dlt:+.0%} points")
            except Exception as ex:
                st.error(f"Could not score this booking: {ex}")

with tab2:
    st.markdown("Upload a CSV with the **same columns as the Kaggle *Hotel Booking Demand* file** "
                "(`hotel, lead_time, arrival_date_*, stays_*, adults, children, babies, meal, country, market_segment, "
                "distribution_channel, is_repeated_guest, previous_*, reserved_room_type, days_in_waiting_list, customer_type, adr, "
                "total_of_special_requests, agent, company`). Outcome columns are ignored.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            df = df.replace("NULL", np.nan)
            probs = pipe.predict_proba(df)[:, 1]
            out = df.copy(); out["cancel_probability"] = probs.round(3); out["risk_tier"] = [tier(p) for p in probs]
            out["recommended_policy"] = out.risk_tier.map({"Low": "No deposit", "Medium": "Partial deposit", "High": "Full deposit / overbooking candidate"})
            c = out.risk_tier.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
            st.bar_chart(c); st.dataframe(out.sort_values("cancel_probability", ascending=False).head(500), use_container_width=True)
            st.download_button("Download scored bookings", out.to_csv(index=False).encode(), "scored_bookings.csv", "text/csv")
        except Exception as ex:
            st.error(f"Could not read/score this file. Check that the columns match the dataset. Details: {ex}")

with tab3:
    m = meta["metrics_test"]; bl = meta["baseline_test"]
    st.markdown(f"**Model:** {meta['model']} · trained on {meta['n_train']:,} bookings, evaluated on {meta['n_test']:,} unseen bookings "
                f"(cancellation rate {meta['cancel_rate']:.1%}).")
    st.table(pd.DataFrame({"Baseline (always 'stays')": bl, "Final model": m}).loc[["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]].round(3))
    st.markdown("""
**How to read it:** of every 100 bookings that really get cancelled, the model flags about 78 in advance; of the bookings it flags, about 6 in 10 really cancel.

**Limitations**
- Data covers two Portuguese hotels, 2015–2017 — patterns may differ for Egyptian hotels or other years; retrain on local data before real use.
- `deposit_type` and post-booking fields (booking changes, parking, assigned room) are intentionally **not** used, to avoid circular or leaky predictions.
- Probabilities support a decision; they do not replace revenue-manager judgement.
""")
