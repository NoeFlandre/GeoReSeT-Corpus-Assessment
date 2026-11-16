# Code quality

Every change must pass `make check`. The tools and thresholds below are the
project requirements.

## Requirements

| Requirement | Tool | Gate |
|---|---|---|
| Formatting and lint | [Ruff](https://docs.astral.sh/ruff/) (`select = ["ALL"]`) | Zero violations |
| Maximum function complexity | Ruff `mccabe` | Complexity ≤ 5 |
| Static types | [ty](https://docs.astral.sh/ty/) | Zero diagnostics |
| No dead code (YAGNI) | [Vulture](https://github.com/jendrikseipp/vulture) | Zero findings at 60% confidence |
| No duplicated code (DRY) | Pylint `duplicate-code` | Zero duplicates of 4+ similar lines |
| Test coverage | Coverage.py, branch coverage | 100% |
| Function risk (CRAP) | `scripts/check_crap.py` | Every function CRAP < 6 |
| Test strength | [mutmut](https://mutmut.readthedocs.io/) | Run and review surviving mutants |
| Documentation builds | MkDocs Material | `mkdocs build --strict` passes |

## CRAP score

For each function, `CRAP = complexity² × (1 − coverage)³ + complexity`, where
complexity is the cyclomatic complexity from Radon and coverage is the fraction
of the function's statements executed by the tests.

A function must stay below 6. With full coverage, that means complexity ≤ 5,
which is why Ruff's `mccabe` limit is also 5. Uncovered complexity is punished
quadratically, so a function with complexity 3 needs at least about 31% of its
statements covered.

## Commands

```bash
make lint        # ruff format/check, ty, vulture, duplicate-code
make test        # pytest with coverage (fails under 100%)
make crap        # CRAP gate, depends on test
make docs        # strict MkDocs build
make mutate      # mutation testing (slow, not part of make check)
make check       # lint, test, crap, docs
```

## Mutation testing

`make mutate` runs mutmut over `src/`. Surviving mutants mean the tests do not
pin down the behaviour. Review them and add tests that kill them. There is no
automatic mutation-score threshold yet.
