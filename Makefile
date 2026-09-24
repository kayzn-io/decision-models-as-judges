.PHONY: check test lint fmt results ui

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy decision_judges
	uv run pytest -q

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

results:
	uv run judges results

ui:
	JUDGES_LOCAL=1 uv run streamlit run decision_judges/ui/app.py
