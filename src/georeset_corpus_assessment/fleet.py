"""Decide which worker jobs to submit or cancel, inside the caps and time windows.

Caps: 24 jobs per site, 60 jobs in total, and no besteffort jobs. Every job of
the account counts, running or waiting. A job waiting longer than 9 minutes
is cancelled. Jobs never cross 09:00 or 19:00 on weekdays, so each job fits
one daytime or one night window.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

CAP_PER_SITE = 24
CAP_TOTAL = 60
WAIT_LIMIT = timedelta(minutes=9)
MAX_WALLTIME_MIN = 30
MARGIN_MIN = 2
LOAD_MIN = 4
MIN_USEFUL_MIN = 15
DAY_START = 9
DAY_END = 19
STRIKES = 2
WAITING_STATES = frozenset({"W", "Waiting"})
FIRST_WEEKEND_DAY = 5
FIELD_PATTERN = re.compile(r"^(\w+) = (.*)$")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class Job:
    """One job of this account, as shown by ``oarstat -f``."""

    job_id: int
    state: str
    submitted: datetime | None


@dataclass
class Health:
    """Consecutive failed workers per site, and the sites currently dropped."""

    failures: dict[str, int] = field(default_factory=dict)

    def dropped(self) -> set[str]:
        """Return the sites with too many failures in a row."""
        return {site for site, count in self.failures.items() if count >= STRIKES}


@dataclass(frozen=True)
class Plan:
    """What to cancel, how many workers to submit per site, and how long they run."""

    cancel: dict[str, list[int]]
    submit: dict[str, int]
    walltime_min: int | None
    budget_min: int | None


def parse_oarstat_full(text: str) -> list[Job]:
    """Parse the output of ``oarstat -f`` into jobs."""
    blocks = [block for block in re.split(r"(?m)^id:", text) if block.strip()]
    return [_job(_fields(block)) for block in blocks]


def _fields(block: str) -> dict[str, str]:
    lines = block.splitlines()
    fields = {"id": lines[0].strip()}
    for line in lines[1:]:
        match = FIELD_PATTERN.match(line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


def _job(fields: dict[str, str]) -> Job:
    raw = fields.get("submission_time", "")
    submitted = datetime.strptime(raw, TIME_FORMAT) if raw else None  # noqa: DTZ007
    return Job(
        job_id=int(fields["id"]), state=fields.get("state", ""), submitted=submitted
    )


def window_end(now: datetime) -> datetime:
    """Return the next boundary: 19:00 in the daytime window, else 09:00."""
    day_window = now.weekday() < FIRST_WEEKEND_DAY and DAY_START <= now.hour < DAY_END
    if day_window:
        return now.replace(hour=DAY_END, minute=0, second=0, microsecond=0)
    morning = now.replace(hour=DAY_START, minute=0, second=0, microsecond=0)
    return morning if now.hour < DAY_START else morning + timedelta(days=1)


def walltime_minutes(now: datetime) -> int | None:
    """Return the longest useful job length that stays inside the window."""
    minutes_left = int((window_end(now) - now).total_seconds() // 60) - MARGIN_MIN
    walltime = min(MAX_WALLTIME_MIN, minutes_left)
    return walltime if walltime >= MIN_USEFUL_MIN else None


def plan(
    jobs_by_site: dict[str, list[Job]],
    now: datetime,
    want: dict[str, int],
    health: Health,
    elsewhere: int = 0,
) -> Plan:
    """Decide the cancels and submissions; ``elsewhere`` counts jobs on other sites."""
    cancel = {site: _stale_waiting(jobs, now) for site, jobs in jobs_by_site.items()}
    active = {
        site: len(jobs) - len(cancel[site]) for site, jobs in jobs_by_site.items()
    }
    walltime = walltime_minutes(now)
    usable = set() if walltime is None else set(want) - health.dropped()
    submit = _submissions(want, active, usable, elsewhere)
    budget = None if walltime is None else walltime - LOAD_MIN - MARGIN_MIN
    return Plan(cancel=cancel, submit=submit, walltime_min=walltime, budget_min=budget)


def _stale_waiting(jobs: list[Job], now: datetime) -> list[int]:
    return [
        job.job_id
        for job in jobs
        if job.state in WAITING_STATES
        and job.submitted is not None
        and now - job.submitted > WAIT_LIMIT
    ]


def _submissions(
    want: dict[str, int],
    active: dict[str, int],
    usable: set[str],
    elsewhere: int,
) -> dict[str, int]:
    total = sum(active.values()) + elsewhere
    submit: dict[str, int] = {}
    for site, wanted in want.items():
        count = 0
        if site in usable:
            room = min(CAP_PER_SITE - active.get(site, 0), CAP_TOTAL - total)
            count = max(0, min(wanted - active.get(site, 0), room))
        submit[site] = count
        total += count
    return submit


def record_finished(health: Health, site: str, *, failed: bool) -> None:
    """Count one finished worker as a failure or as a success."""
    health.failures[site] = health.failures.get(site, 0) + 1 if failed else 0


def record_running(health: Health, site: str) -> None:
    """Clear the failure count and any drop of a site whose job runs again."""
    health.failures[site] = 0


def load_health(text: str) -> Health:
    """Read the saved health state; an empty text means no history."""
    return Health(failures=json.loads(text) if text.strip() else {})


def dump_health(health: Health) -> str:
    """Serialise the health state."""
    return json.dumps(health.failures, sort_keys=True)


def apply_outcomes(
    health: Health,
    site: str,
    *,
    failures: int,
    successes: int,
    running: bool,
) -> None:
    """Apply the outcomes seen since the last check to one site's health."""
    for _ in range(failures):
        record_finished(health, site, failed=True)
    for _ in range(successes):
        record_finished(health, site, failed=False)
    if running:
        record_running(health, site)


def render_actions(decision: Plan, site: str) -> list[str]:
    """Return one command line per action: cancel IDs, then one submit line."""
    lines = [f"cancel {job_id}" for ids in decision.cancel.values() for job_id in ids]
    if _can_submit(decision, site):
        count = decision.submit[site]
        lines.append(f"submit {count} {decision.walltime_min} {decision.budget_min}")
    return lines


def _can_submit(decision: Plan, site: str) -> bool:
    return bool(decision.submit.get(site, 0)) and decision.walltime_min is not None
