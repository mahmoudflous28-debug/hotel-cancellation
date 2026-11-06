"""Shared feature engineering for the hotel cancellation project.
Kept in its own module so the saved joblib pipeline can be loaded by the Streamlit app."""
import numpy as np
import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Columns that are NOT known at booking time (or reveal the outcome) -> never used
LEAKAGE_COLS = ["reservation_status", "reservation_status_date",
                "assigned_room_type", "booking_changes", "required_car_parking_spaces"]
# Excluded on purpose: deposit_type is the very policy we want to recommend (circular),
# and "Non Refund" is a data artefact (~95% cancelled).
EXCLUDED_COLS = ["deposit_type"]

NUM_FEATURES = ["lead_time", "total_nights", "weekend_share", "total_guests", "adults",
                "children", "babies", "is_family", "previous_cancellations",
                "previous_bookings_not_canceled", "is_repeated_guest", "days_in_waiting_list",
                "adr", "adr_per_person",
                "total_of_special_requests", "has_special_requests", "has_agent", "has_company",
                "arrival_month_num", "arrival_week", "arrival_dow",
                "month_sin", "month_cos"]
CAT_FEATURES = ["hotel", "meal", "country", "market_segment", "distribution_channel",
                "reserved_room_type", "customer_type"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw booking columns -> model features. Only uses information known before arrival."""
    d = df.copy()
    d["children"] = pd.to_numeric(d["children"], errors="coerce").fillna(0)
    d["adr"] = pd.to_numeric(d["adr"], errors="coerce").clip(lower=0, upper=1000)
    d["country"] = d["country"].fillna("Unknown").replace("NULL", "Unknown")

    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["weekend_share"] = np.where(d["total_nights"] > 0,
                                  d["stays_in_weekend_nights"] / d["total_nights"].replace(0, np.nan), 0)
    d["total_guests"] = d["adults"] + d["children"] + d["babies"]
    d["is_family"] = ((d["children"] + d["babies"]) > 0).astype(int)
    d["adr_per_person"] = d["adr"] / d["total_guests"].replace(0, np.nan)
    d["has_special_requests"] = (d["total_of_special_requests"] > 0).astype(int)
    d["has_agent"] = d["agent"].notna().astype(int) if d["agent"].dtype != object else \
        (~d["agent"].isin(["NULL", None, np.nan])).astype(int)
    d["has_company"] = d["company"].notna().astype(int) if d["company"].dtype != object else \
        (~d["company"].isin(["NULL", None, np.nan])).astype(int)

    m = d["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    d["arrival_month_num"] = m
    d["arrival_week"] = d["arrival_date_week_number"]
    dt = pd.to_datetime(dict(year=d["arrival_date_year"], month=m, day=d["arrival_date_day_of_month"]),
                        errors="coerce")
    d["arrival_dow"] = dt.dt.dayofweek
    d["month_sin"] = np.sin(2 * np.pi * m / 12)
    d["month_cos"] = np.cos(2 * np.pi * m / 12)
    return d[NUM_FEATURES + CAT_FEATURES]
