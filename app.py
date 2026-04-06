import io
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from boto3.dynamodb.conditions import Key

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NOAA_API     = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION_ID   = "8638610"       # Sewells Point, Norfolk VA
STATION_NAME = "Sewells Point, Norfolk VA"
TABLE_NAME   = os.environ["DYNAMODB_TABLE"]
S3_BUCKET    = os.environ["S3_BUCKET"]
AWS_REGION   = os.environ.get("AWS_REGION", "us-east-1")

# Water level this many feet above the predicted level is flagged as storm surge.
# NOAA defines minor coastal flooding at ~1.5 ft above predicted; we use 1.0 ft
# as an early-warning threshold.
SURGE_THRESHOLD_FT = Decimal("1.0")


# ---------------------------------------------------------------------------
# Step 1 — Fetch latest observed + predicted water level from NOAA
# ---------------------------------------------------------------------------
def fetch_water_level() -> dict:
    """Return a DynamoDB-ready item with the current water level state."""

    def _get(product: str) -> float:
        resp = requests.get(
            NOAA_API,
            params={
                "date":      "latest",
                "station":   STATION_ID,
                "product":   product,
                "datum":     "MLLW",
                "time_zone": "gmt",
                "units":     "english",
                "format":    "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise ValueError(f"NOAA API error for {product}: {data['error']['message']}")
        return float(data["data"][0]["v"])

    observed  = _get("water_level")
    predicted = _get("predictions")
    now_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "station_id":    STATION_ID,
        "timestamp":     now_ts,
        "observed_ft":   Decimal(str(round(observed,  3))),
        "predicted_ft":  Decimal(str(round(predicted, 3))),
        "surge_ft":      Decimal(str(round(observed - predicted, 3))),
    }


# ---------------------------------------------------------------------------
# Step 2 — Query DynamoDB for the most recent previous entry
# ---------------------------------------------------------------------------
def get_previous(table) -> dict | None:
    """Return the latest stored item for this station, or None on first run."""
    resp = table.query(
        KeyConditionExpression=Key("station_id").eq(STATION_ID),
        ScanIndexForward=False,
        Limit=1,
    )
    items = resp.get("Items", [])
    return items[0] if items else None


# ---------------------------------------------------------------------------
# Step 3 — Classify the current water level event
# ---------------------------------------------------------------------------
def classify_event(entry: dict, previous: dict | None) -> str:
    """Return a trend/event label for the current reading.

    Labels:
      FIRST_ENTRY  — no prior data
      STORM_SURGE  — observed is >= SURGE_THRESHOLD_FT above predicted
      ABOVE_PRED   — observed moderately above predicted (0–SURGE_THRESHOLD_FT)
      NEAR_PRED    — observed close to predicted (+/- 0.1 ft)
      BELOW_PRED   — observed below predicted
    """
    if previous is None:
        return "FIRST_ENTRY"

    surge = entry["surge_ft"]

    if surge >= SURGE_THRESHOLD_FT:
        return "STORM_SURGE"
    elif surge > Decimal("0.1"):
        return "ABOVE_PRED"
    elif surge >= Decimal("-0.1"):
        return "NEAR_PRED"
    else:
        return "BELOW_PRED"


# ---------------------------------------------------------------------------
# Step 4 — Fetch full history from DynamoDB for plotting
# ---------------------------------------------------------------------------
def fetch_history(table) -> pd.DataFrame:
    """Return all stored records as a DataFrame, sorted by timestamp."""
    items, kwargs = [], dict(
        KeyConditionExpression=Key("station_id").eq(STATION_ID),
        ScanIndexForward=True,
    )
    while True:
        resp = table.query(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["observed_ft"]  = df["observed_ft"].astype(float)
    df["predicted_ft"] = df["predicted_ft"].astype(float)
    df["surge_ft"]     = df["surge_ft"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 — Render water level plot
# ---------------------------------------------------------------------------
def generate_plot(df: pd.DataFrame) -> io.BytesIO | None:
    """Plot observed vs predicted water level over time with surge annotations."""
    if df.empty or len(df) < 2:
        log.info("Not enough history to plot yet (%d point(s))", len(df))
        return None

    sns.set_theme(style="darkgrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Observed water level
    sns.lineplot(data=df, x="timestamp", y="observed_ft",
                 ax=ax, color="#4FC3F7", linewidth=2.5, label="Observed", zorder=3)

    # Predicted tide line
    sns.lineplot(data=df, x="timestamp", y="predicted_ft",
                 ax=ax, color="#B0BEC5", linewidth=1.5, linestyle="--",
                 label="Predicted", zorder=2)

    # Shade area between observed and predicted
    ax.fill_between(
        df["timestamp"],
        df["observed_ft"],
        df["predicted_ft"],
        where=df["observed_ft"] >= df["predicted_ft"],
        alpha=0.20, color="#FF6B35", label="Above predicted",
    )
    ax.fill_between(
        df["timestamp"],
        df["observed_ft"],
        df["predicted_ft"],
        where=df["observed_ft"] < df["predicted_ft"],
        alpha=0.12, color="#4FC3F7",
    )

    # Highlight storm surge events
    surges = df[df["surge_ft"] >= float(SURGE_THRESHOLD_FT)]
    if not surges.empty:
        ax.scatter(surges["timestamp"], surges["observed_ft"],
                   color="#FF6B35", s=140, zorder=5,
                   label=f"Storm surge ({len(surges)} events)")
        for _, row in surges.iterrows():
            ax.annotate(
                "🌊",
                xy=(row["timestamp"], row["observed_ft"]),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center", fontsize=16, zorder=6,
            )

    ax.set_title(
        f"Water Level — {STATION_NAME}\n"
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Time (UTC)", labelpad=8)
    ax.set_ylabel("Water Level (ft, MLLW)", labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} ft"))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85, edgecolor="#555555")

    sns.despine(ax=ax, top=True, right=True)
    fig.autofmt_xdate(rotation=25, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    log.info("Plot generated (%d bytes, %d points)", len(buf.getvalue()), len(df))
    return buf


# ---------------------------------------------------------------------------
# Step 6 — Upload plot to S3
# ---------------------------------------------------------------------------
def push_plot(buf: io.BytesIO) -> None:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="plot.png",
        Body=buf.getvalue(),
        ContentType="image/png",
    )
    log.info("Uploaded plot.png to s3://%s", S3_BUCKET)


# ---------------------------------------------------------------------------
# Step 7 — Upload CSV snapshot to S3
# ---------------------------------------------------------------------------
def push_csv(df: pd.DataFrame) -> None:
    s3  = boto3.client("s3", region_name=AWS_REGION)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="data.csv",
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    log.info("Uploaded data.csv to s3://%s (%d rows)", S3_BUCKET, len(df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table    = dynamodb.Table(TABLE_NAME)

    previous = get_previous(table)
    entry    = fetch_water_level()
    event    = classify_event(entry, previous)

    entry["event"] = event
    table.put_item(Item=entry)

    surge_flag = "  *** STORM SURGE DETECTED ***" if event == "STORM_SURGE" else ""
    log.info(
        "TIDE | observed=%.3f ft | predicted=%.3f ft | surge=%+.3f ft | %-11s | station=%s%s",
        entry["observed_ft"],
        entry["predicted_ft"],
        entry["surge_ft"],
        event,
        STATION_ID,
        surge_flag,
    )

    history  = fetch_history(table)
    plot_buf = generate_plot(history)
    if plot_buf:
        push_plot(plot_buf)
        push_csv(history)


if __name__ == "__main__":
    main()