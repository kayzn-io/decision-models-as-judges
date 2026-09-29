"""Judge a chosen sample of uncut full copies with gpt-5, writing to the shared cache.

Usage: OPENROUTER_API_KEY=... uv run python scripts/sample_strong.py

The sample is 12 conversations the state check passed but gpt-5 failed on all
five repeats on the cut copies, every time citing unsupported facts, plus 3
conversations that truly failed and gpt-5 passed on all five. If the cap was
the cause, the first group should flip; the second should not.
"""

from pathlib import Path

from decision_judges import pipeline
from decision_judges.bench.load import load_tasks
from decision_judges.cache import Cache
from decision_judges.config import load_pricing, load_study
from decision_judges.serialize import StateProfile
from decision_judges.spend import Spend

ROOT = Path(__file__).resolve().parents[1]

FLIP_CANDIDATES = [
    ("baseline", "retail-23"),
    ("baseline", "retail-95"),
    ("baseline", "retail-0"),
    ("baseline", "retail-48"),
    ("baseline", "retail-46"),
    ("baseline", "retail-10"),
    ("degraded", "retail-93"),
    ("degraded", "retail-16"),
    ("degraded", "retail-95"),
    ("degraded", "retail-92"),
    ("degraded", "retail-35"),
    ("degraded", "retail-63"),
]
CONTROLS = [("baseline", "retail-58"), ("baseline", "retail-4"), ("degraded", "retail-32")]
REPEATS = 5


def main() -> None:
    study = load_study(ROOT / "config/study.toml")
    pricing = load_pricing(ROOT / "config/pricing.toml")
    tasks = {t.task_id: t for t in load_tasks()}
    gate = pipeline.make_gate("g3", tasks)
    cache = Cache(ROOT / "cache/judge")
    spend = Spend(pricing, study.spend_caps, ROOT / "results/spend.json")

    wanted = set(FLIP_CANDIDATES + CONTROLS)
    for variant in ["baseline", "degraded"]:
        records, _ = pipeline.load_agent_records(ROOT / "cache/agent", variant)
        states = pipeline.read_states(ROOT / "cache/state", variant, StateProfile.full)
        chosen = [s for t, s in states.items() if (variant, t) in wanted]
        items = pipeline.items_from_states(chosen, records)
        specs = pipeline.judge_specs_from(["llm_strong"], study)
        judges = pipeline.build_judges(
            specs,
            study=study,
            tasks=tasks,
            records=records,
            rubric_path=pipeline.gate_rubric_path(gate),
            prompt_version=pipeline.gate_prompt_version(gate),
        )
        verdicts = pipeline.run_gate(gate, items, judges, cache, spend, REPEATS)
        print(f"{variant}: {len(items)} conversations, {len(verdicts)} verdicts")
    print(f"g3 spend so far: ${spend.spent('g3'):.4f}")


if __name__ == "__main__":
    main()
