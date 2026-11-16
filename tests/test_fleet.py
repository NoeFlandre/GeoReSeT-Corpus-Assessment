from datetime import datetime, timedelta

from georeset_corpus_assessment.fleet import (
    Health,
    Job,
    Plan,
    apply_outcomes,
    dump_health,
    load_health,
    parse_oarstat_full,
    plan,
    record_finished,
    record_running,
    render_actions,
    walltime_minutes,
    window_end,
)

MONDAY_1800 = datetime(2026, 11, 16, 18, 0)  # noqa: DTZ001
MONDAY_2300 = datetime(2026, 11, 16, 23, 0)  # noqa: DTZ001
MONDAY_0700 = datetime(2026, 11, 16, 7, 0)  # noqa: DTZ001
MONDAY_1000 = datetime(2026, 11, 16, 10, 0)  # noqa: DTZ001

OARSTAT = """id: 2230781
    name = georeset-dspark-1000-6h
    state = Hold
    submission_time = 2026-10-12 15:03:06
2026-10-12 15:11:10> HOLD_WAITING_JOB: User nflandre launched oarhold on the job 2230781
id: 2249104
    name = lille-u0
    state = W
    submission_time = 2026-11-16 18:00:00
"""


def test_window_end_is_19h_in_daytime_and_9h_at_night() -> None:
    assert window_end(MONDAY_1000) == datetime(2026, 11, 16, 19, 0)  # noqa: DTZ001
    assert window_end(MONDAY_2300) == datetime(2026, 11, 17, 9, 0)  # noqa: DTZ001
    assert window_end(MONDAY_0700) == datetime(2026, 11, 16, 9, 0)  # noqa: DTZ001


def test_walltime_is_capped_at_an_hour_and_none_when_too_short() -> None:
    assert walltime_minutes(MONDAY_2300) == 30
    assert walltime_minutes(datetime(2026, 11, 16, 18, 50)) is None  # noqa: DTZ001
    assert walltime_minutes(datetime(2026, 11, 16, 18, 30)) == 28  # noqa: DTZ001


def test_parse_reads_each_job_and_its_submission_time() -> None:
    jobs = parse_oarstat_full(OARSTAT)
    assert [job.job_id for job in jobs] == [2230781, 2249104]
    assert jobs[1].state == "W"
    assert jobs[1].submitted == datetime(2026, 11, 16, 18, 0)  # noqa: DTZ001


def test_plan_cancels_waiting_jobs_older_than_nine_minutes() -> None:
    old = Job(1, "W", MONDAY_1800 - timedelta(minutes=10))
    fresh = Job(2, "W", MONDAY_1800 - timedelta(minutes=2))
    running = Job(3, "R", MONDAY_1800 - timedelta(hours=1))
    result = plan({"lille": [old, fresh, running]}, MONDAY_1800, {}, Health())
    assert result.cancel == {"lille": [1]}


def test_plan_respects_site_and_total_caps() -> None:
    lille = [Job(i, "R", MONDAY_1800) for i in range(22)]
    nantes = [Job(100 + i, "R", MONDAY_1800) for i in range(30)]
    result = plan(
        {"lille": lille, "nantes": nantes},
        MONDAY_1000,
        {"lille": 30, "nantes": 30},
        Health(),
    )
    assert result.submit == {"lille": 2, "nantes": 0}


def test_plan_submits_nothing_when_the_window_is_too_short() -> None:
    result = plan({"lille": []}, datetime(2026, 11, 16, 18, 55), {"lille": 5}, Health())  # noqa: DTZ001
    assert result.submit == {"lille": 0}
    assert result.walltime_min is None
    assert result.budget_min is None


def test_two_failures_in_a_row_drop_a_site_until_a_job_runs_again() -> None:
    health = Health()
    record_finished(health, "lille", failed=True)
    assert health.dropped() == set()
    record_finished(health, "lille", failed=True)
    assert health.dropped() == {"lille"}
    result = plan({"lille": []}, MONDAY_1000, {"lille": 4}, health)
    assert result.submit == {"lille": 0}
    record_running(health, "lille")
    assert health.dropped() == set()


def test_a_success_resets_the_failure_count() -> None:
    health = Health()
    record_finished(health, "nantes", failed=True)
    record_finished(health, "nantes", failed=False)
    assert health.failures == {"nantes": 0}


def test_health_round_trips_through_text() -> None:
    text = dump_health(Health(failures={"lille": 2}))
    assert load_health(text).dropped() == {"lille"}
    assert load_health("").failures == {}


def test_apply_outcomes_counts_failures_then_clears_on_running() -> None:
    health = Health()
    apply_outcomes(health, "lille", failures=2, successes=0, running=False)
    assert health.dropped() == {"lille"}
    apply_outcomes(health, "lille", failures=0, successes=0, running=True)
    assert health.dropped() == set()


def test_render_actions_lists_cancels_then_one_submit_line() -> None:
    decision = Plan(
        cancel={"lille": [7, 8]}, submit={"lille": 3}, walltime_min=40, budget_min=34
    )
    assert render_actions(decision, "lille") == [
        "cancel 7",
        "cancel 8",
        "submit 3 40 34",
    ]


def test_render_actions_skips_submit_when_no_window_or_nothing_to_submit() -> None:
    no_window = Plan(cancel={}, submit={"lille": 3}, walltime_min=None, budget_min=None)
    nothing = Plan(cancel={}, submit={"lille": 0}, walltime_min=40, budget_min=34)
    assert render_actions(no_window, "lille") == []
    assert render_actions(nothing, "lille") == []


def test_plan_counts_jobs_elsewhere_against_the_total_cap() -> None:
    jobs = [Job(i, "R", MONDAY_1000) for i in range(2)]
    result = plan({"lille": jobs}, MONDAY_1000, {"lille": 20}, Health(), elsewhere=57)
    assert result.submit == {"lille": 1}


def test_apply_outcomes_records_successes_too() -> None:
    health = Health()
    apply_outcomes(health, "nantes", failures=0, successes=2, running=False)
    assert health.failures == {"nantes": 0}


def test_long_state_names_from_full_oarstat_count_as_waiting() -> None:
    old = Job(5, "Waiting", MONDAY_1800 - timedelta(minutes=20))
    result = plan({"lille": [old]}, MONDAY_1800, {}, Health())
    assert result.cancel == {"lille": [5]}


def test_want_is_a_target_not_an_increment() -> None:
    jobs = [Job(i, "W", MONDAY_1000) for i in range(6)]
    result = plan({"nantes": jobs}, MONDAY_1000, {"nantes": 6}, Health())
    assert result.submit == {"nantes": 0}
