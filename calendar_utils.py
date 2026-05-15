"""Trading-calendar utilities for backtests and live-signal style workflows."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_CALENDAR_PATH = BASE_DIR / "data" / "trading_calendar_2026.csv"
BERLIN_TZ = ZoneInfo("Europe/Berlin")


def get_trading_days(prices_or_returns: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    """Return sorted trading days from a price or return object."""

    index = pd.to_datetime(prices_or_returns.index)
    return pd.DatetimeIndex(index).sort_values().unique()


def is_last_trading_day_of_week(
    date: pd.Timestamp | str,
    trading_days: pd.DatetimeIndex,
) -> bool:
    """Return whether the supplied date is the final available trading day of its ISO week."""

    day = pd.Timestamp(date)
    normalized_days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values().unique()
    week_days = normalized_days[
        (normalized_days.isocalendar().year == day.isocalendar().year)
        & (normalized_days.isocalendar().week == day.isocalendar().week)
    ]
    if len(week_days) == 0:
        return False
    return pd.Timestamp(week_days[-1]) == day


def get_next_trading_day(
    date: pd.Timestamp | str,
    trading_days: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Return the next trading day after the supplied date."""

    day = pd.Timestamp(date)
    normalized_days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values().unique()
    future_days = normalized_days[normalized_days > day]
    if len(future_days) == 0:
        return None
    return pd.Timestamp(future_days[0])


def get_previous_trading_day(
    date: pd.Timestamp | str,
    trading_days: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Return the previous trading day before the supplied date."""

    day = pd.Timestamp(date)
    normalized_days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values().unique()
    past_days = normalized_days[normalized_days < day]
    if len(past_days) == 0:
        return None
    return pd.Timestamp(past_days[-1])


def ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized DatetimeIndex."""

    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def latest_trading_day(prices: pd.DataFrame) -> pd.Timestamp:
    """Return the final trading day present in the input history."""

    trading_days = get_trading_days(prices)
    if len(trading_days) == 0:
        raise ValueError("Price history is empty.")
    return pd.Timestamp(trading_days[-1])


def get_expected_latest_trading_day(
    now: datetime | pd.Timestamp | None = None,
    calendar_path: str | Path = DEFAULT_PROJECT_CALENDAR_PATH,
) -> str | None:
    """Return the latest fully completed project trading day expected for current-run reports.

    The daily bot and research reports work off completed daily bars. For a run on a
    trading day, the expected latest price date is therefore the previous project
    trading day rather than the still-in-progress current session.
    """

    current_berlin = _coerce_berlin_datetime(now)
    current_date = current_berlin.date()
    calendar_df = load_trading_calendar(calendar_path)
    trading_days = calendar_df.loc[calendar_df["is_trading_day"], "date"].sort_values()
    if trading_days.empty:
        return None

    if is_project_trading_day(current_date, calendar_df):
        eligible_days = trading_days.loc[trading_days < current_date]
    else:
        eligible_days = trading_days.loc[trading_days <= current_date]

    if eligible_days.empty:
        return None
    return pd.Timestamp(eligible_days.iloc[-1]).date().isoformat()


def previous_trading_day(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    """Compatibility alias for get_previous_trading_day()."""

    return get_previous_trading_day(date=date, trading_days=index)


def get_backtest_dates(prices: pd.DataFrame, warmup_days: int) -> pd.DatetimeIndex:
    """Return evaluation dates after a warmup period."""

    trading_days = get_trading_days(prices)
    if len(trading_days) <= warmup_days:
        raise ValueError("Not enough observations for the configured warmup period.")
    return trading_days[warmup_days:]


def load_trading_calendar(path: str | Path = DEFAULT_PROJECT_CALENDAR_PATH) -> pd.DataFrame:
    """Load the local project trading calendar."""

    calendar_path = Path(path)
    calendar_df = pd.read_csv(calendar_path)
    required_columns = {
        "date",
        "is_trading_day",
        "allowed_start_berlin",
        "allowed_end_berlin",
        "reason",
    }
    missing = required_columns.difference(calendar_df.columns)
    if missing:
        raise ValueError(
            f"Trading calendar is missing required columns: {', '.join(sorted(missing))}"
        )
    calendar_df = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.date
    calendar_df["is_trading_day"] = calendar_df["is_trading_day"].astype(bool)
    calendar_df["allowed_start_berlin"] = calendar_df["allowed_start_berlin"].fillna("").astype(str)
    calendar_df["allowed_end_berlin"] = calendar_df["allowed_end_berlin"].fillna("").astype(str)
    calendar_df["reason"] = calendar_df["reason"].fillna("").astype(str)
    return calendar_df.sort_values("date").reset_index(drop=True)


def is_project_trading_day(date: pd.Timestamp | datetime | str, calendar_df: pd.DataFrame) -> bool:
    """Return whether the supplied date is a project trading day."""

    day = pd.Timestamp(date).date()
    row = calendar_df.loc[calendar_df["date"] == day]
    if row.empty:
        return False
    return bool(row.iloc[0]["is_trading_day"])


def get_allowed_window_berlin(
    date: pd.Timestamp | datetime | str,
    calendar_df: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """Return the allowed Berlin trading window for the supplied date."""

    day = pd.Timestamp(date).date()
    row = calendar_df.loc[calendar_df["date"] == day]
    if row.empty:
        return None, None
    start = str(row.iloc[0]["allowed_start_berlin"]).strip() or None
    end = str(row.iloc[0]["allowed_end_berlin"]).strip() or None
    return start, end


def _coerce_berlin_datetime(now: datetime | pd.Timestamp | None = None) -> datetime:
    """Return a timezone-aware Berlin datetime for market-gate checks."""

    if now is None:
        return datetime.now(BERLIN_TZ)
    if isinstance(now, pd.Timestamp):
        now = now.to_pydatetime()
    if now.tzinfo is None:
        return now.replace(tzinfo=BERLIN_TZ)
    return now.astimezone(BERLIN_TZ)


def _parse_hhmm(value: str | None) -> time | None:
    """Parse a HH:MM string into a time object."""

    text = str(value or "").strip()
    if not text:
        return None
    return datetime.strptime(text, "%H:%M").time()


def is_within_project_trading_window(
    now: datetime | pd.Timestamp | None = None,
    calendar_path: str | Path = DEFAULT_PROJECT_CALENDAR_PATH,
) -> dict[str, object]:
    """Return whether execution is currently allowed by the local project calendar."""

    current_berlin = _coerce_berlin_datetime(now)
    current_date = current_berlin.date()
    current_time = current_berlin.time().replace(second=0, microsecond=0)
    calendar_df = load_trading_calendar(calendar_path)
    row = calendar_df.loc[calendar_df["date"] == current_date]

    if row.empty:
        return {
            "date": current_date.isoformat(),
            "current_time_berlin": current_time.strftime("%H:%M"),
            "is_trading_day": False,
            "within_allowed_window": False,
            "allowed_start_berlin": None,
            "allowed_end_berlin": None,
            "execution_allowed": False,
            "reason": "date_not_in_project_calendar",
        }

    row_data = row.iloc[0]
    allowed_start = str(row_data["allowed_start_berlin"]).strip() or None
    allowed_end = str(row_data["allowed_end_berlin"]).strip() or None
    is_trading_day = bool(row_data["is_trading_day"])
    if not is_trading_day:
        return {
            "date": current_date.isoformat(),
            "current_time_berlin": current_time.strftime("%H:%M"),
            "is_trading_day": False,
            "within_allowed_window": False,
            "allowed_start_berlin": allowed_start,
            "allowed_end_berlin": allowed_end,
            "execution_allowed": False,
            "reason": str(row_data["reason"] or "non_trading_day"),
        }

    start_time = _parse_hhmm(allowed_start)
    end_time = _parse_hhmm(allowed_end)
    within_window = (
        start_time is not None
        and end_time is not None
        and start_time <= current_time <= end_time
    )
    return {
        "date": current_date.isoformat(),
        "current_time_berlin": current_time.strftime("%H:%M"),
        "is_trading_day": True,
        "within_allowed_window": within_window,
        "allowed_start_berlin": allowed_start,
        "allowed_end_berlin": allowed_end,
        "execution_allowed": bool(within_window),
        "reason": "within_project_trading_window" if within_window else "outside_allowed_window",
    }


def _smoke_test_cases() -> list[tuple[datetime, bool, str]]:
    """Return fixed smoke-test cases for the project calendar."""

    return [
        (datetime(2026, 4, 24, 16, 30, tzinfo=BERLIN_TZ), True, "within_project_trading_window"),
        (datetime(2026, 4, 24, 15, 45, tzinfo=BERLIN_TZ), False, "outside_allowed_window"),
        (datetime(2026, 5, 25, 17, 0, tzinfo=BERLIN_TZ), False, "Memorial Day"),
        (datetime(2026, 6, 20, 17, 0, tzinfo=BERLIN_TZ), False, "weekend"),
        (datetime(2026, 7, 24, 21, 30, tzinfo=BERLIN_TZ), True, "within_project_trading_window"),
        (datetime(2026, 7, 24, 22, 30, tzinfo=BERLIN_TZ), False, "outside_allowed_window"),
    ]


def _run_smoke_tests() -> int:
    """Run simple project-calendar smoke tests."""

    failures: list[str] = []
    for moment, expected_allowed, expected_reason in _smoke_test_cases():
        result = is_within_project_trading_window(now=moment)
        if bool(result["execution_allowed"]) != expected_allowed or str(result["reason"]) != expected_reason:
            failures.append(
                f"{moment.isoformat()} -> expected ({expected_allowed}, {expected_reason}) "
                f"but got ({result['execution_allowed']}, {result['reason']})"
            )
    if failures:
        for failure in failures:
            print(failure)
        return 1
    for moment, _, _ in _smoke_test_cases():
        print(is_within_project_trading_window(now=moment))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_smoke_tests())
