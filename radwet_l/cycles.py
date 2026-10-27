"""ALOS-2 PALSAR-2 ScanSAR orbit-cycle calendar.

ALOS-2 repeats its orbit every 14 days. RadWet-L products are named by this
cycle number (e.g. Zenodo's ``Classified_Output_Orbit-Cycle_226-...``). The
calendar in ``data/cycle_dates.csv`` comes from the original repository.
"""
import csv
import datetime as dt
from functools import lru_cache

from .config import CYCLE_DATES_CSV


@lru_cache(maxsize=1)
def load_cycles():
    """Return {cycle: (start_date, end_date)}; both dates inclusive."""
    cycles = {}
    with open(CYCLE_DATES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            start = dt.datetime.strptime(row["Start"].strip(), "%d/%m/%y").date()
            end = dt.datetime.strptime(row["End"].strip(), "%d/%m/%y").date()
            cycles[int(row["Cycle"])] = (start, end)
    return cycles


def cycle_dates(cycle):
    """Return (start, end) dates of an orbit cycle, both inclusive."""
    cycles = load_cycles()
    if cycle not in cycles:
        raise ValueError(
            f"Cycle {cycle} is not in {CYCLE_DATES_CSV.name} "
            f"(covers {min(cycles)}-{max(cycles)})"
        )
    return cycles[cycle]


def cycle_for_date(date):
    """Return the orbit cycle containing ``date`` (date or YYYY-MM-DD), or None."""
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    for cycle, (start, end) in load_cycles().items():
        if start <= date <= end:
            return cycle
    return None
