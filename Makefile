.PHONY: sync format lint typecheck dead-code duplicates test crap docs mutate check

sync:
	uv sync

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff format --check
	uv run ruff check
	uv run pylint --disable=all --enable=duplicate-code src tests scripts
	uv run vulture

typecheck:
	uv run ty check

dead-code:
	uv run vulture

duplicates:
	uv run pylint --disable=all --enable=duplicate-code src tests scripts

test:
	uv run coverage run -m pytest
	uv run coverage report
	uv run coverage json -o coverage.json

crap: test
	uv run python scripts/check_crap.py coverage.json

docs:
	uv run mkdocs build --strict

mutate:
	uv run mutmut run

check: lint typecheck test crap docs
