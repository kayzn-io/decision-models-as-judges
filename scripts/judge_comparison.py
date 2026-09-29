"""Print every table in docs/judge-comparison.md from the committed caches.

Reads cache/agent, cache/state, and cache/judge, keeps the four judges that read
the full reading copy, takes the modal verdict of the five repeats per judge and
conversation, and prints the comparison tables. No paid call is made.

    uv run python scripts/judge_comparison.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from decision_judges import pipeline
from decision_judges.bench.load import load_tasks, normalize_action
from decision_judges.judges.code import _tool_calls
from decision_judges.metrics import accuracy, auroc, brier, cohens_kappa, ece, precision_recall_f1
from decision_judges.serialize import StateProfile

ROOT = Path(__file__).resolve().parents[1]
JUDGES = ["code", "jev", "llm_cheap", "llm_strong"]
MODELS = ["jev", "llm_cheap", "llm_strong"]
VARIANTS = ["baseline", "degraded"]
PRICE_PER_MTOK = {
    "code": (0.0, 0.0),
    "jev": (0.042, 0.0),
    "llm_cheap": (0.15, 0.60),
    "llm_strong": (1.25, 10.0),
}
READ_PREFIXES = ("get_", "find_", "list_", "calculate", "think", "transfer")


def show(title: str, frame: pd.DataFrame) -> None:
    print(f"\n## {title}\n")
    print(frame.to_markdown())


def load_items() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        records, _ = pipeline.load_agent_records(ROOT / "cache/agent", variant)
        states = pipeline.read_states(ROOT / "cache/state", variant, StateProfile.full)
        for item in pipeline.items_from_states(list(states.values()), records):
            record = records[item.state.task_id]
            rows.append(
                {
                    "state_hash": item.state.state_hash,
                    "task_id": item.state.task_id,
                    "variant": variant,
                    "truth": item.truth_label,
                    "n_tool_calls": sum(
                        len(m.get("tool_calls") or [])
                        for m in record.trajectory
                        if m.get("role") == "assistant"
                    ),
                    "truncated_results": item.state.text.lower().count("[truncated"),
                    "tool_results": len(re.findall(r"^tool:", item.state.text, re.M)),
                }
            )
    return pd.DataFrame(rows)


def load_verdicts(items: pd.DataFrame) -> pd.DataFrame:
    hashes = set(items.state_hash)
    verdicts, _ = pipeline.load_verdicts(ROOT / "cache/judge", judge_ids=set(JUDGES))
    rows = []
    seen: set[tuple[str, str, int]] = set()
    for v in verdicts:
        key = (v.judge_id, v.state_hash, v.repeat)
        if v.state_hash not in hashes or key in seen:
            continue
        seen.add(key)
        verdict = next((a for a in v.answers if a.question_id == "verdict"), None)
        completed = next((a.noul for a in v.answers if a.question_id == "completed"), None)
        rows.append(
            {
                "judge": v.judge_id,
                "state_hash": v.state_hash,
                "repeat": v.repeat,
                "choice": verdict.choice if verdict else None,
                "p_pass": (verdict.probabilities or {}).get("pass") if verdict else None,
                "confidence": verdict.confidence if verdict else None,
                "completed": completed,
                "input_tokens": v.usage.input_tokens,
                "output_tokens": v.usage.output_tokens,
                "latency_ms": v.latency_ms,
                "rationale": v.rationale or "",
            }
        )
    return pd.DataFrame(rows).merge(items, on="state_hash")


def modal(choices: pd.Series) -> str:
    counts = choices.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def aggregate(verdicts: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    return (
        verdicts.groupby(["judge", "state_hash"])
        .agg(
            modal=("choice", modal),
            p_pass=("p_pass", "mean"),
            completed=("completed", "mean"),
            confidence=("confidence", "mean"),
            n_pass=("choice", lambda s: int((s == "pass").sum())),
            unanimous=("choice", lambda s: s.nunique() == 1),
        )
        .reset_index()
        .merge(items, on="state_hash")
    )


def summarize(sub: pd.DataFrame) -> dict[str, float]:
    truth = sub.truth.tolist()
    pred = sub.modal.tolist()
    t01 = [1.0 if t == "pass" else 0.0 for t in truth]
    fail = precision_recall_f1(pred, truth, "fail")
    return {
        "n": len(sub),
        "accuracy": accuracy(pred, truth),
        "kappa": cohens_kappa(pred, truth),
        "auroc_p_pass": auroc(sub.p_pass.tolist(), t01),
        "auroc_completed": auroc(sub.completed.tolist(), t01),
        "recall_fail": fail.recall,
        "precision_fail": fail.precision,
        "pass_rate": float(np.mean([p == "pass" for p in pred])),
        "unanimous": float(sub.unanimous.mean()),
        "ece_p_pass": ece(sub.p_pass.tolist(), t01),
        "brier_p_pass": brier(sub.p_pass.tolist(), t01),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def miss_kind(variant: str, task_id: str, tasks: dict) -> str:
    records, _ = pipeline.load_agent_records(ROOT / "cache/agent", variant)
    record = records[task_id]
    task = tasks[task_id]
    expected = {
        e
        for e in (normalize_action(a.name, a.kwargs) for a in task.actions)
        if not e[0].startswith(READ_PREFIXES)
    }
    made = {m for m in _tool_calls(record) if not m[0].startswith(READ_PREFIXES)}
    if expected == made:
        text = "\n".join(
            str(m.get("content") or "") for m in record.trajectory if m.get("role") == "assistant"
        ).lower()
        outputs_ok = all(o.lower() in text for o in task.outputs)
        return "other" if outputs_ok else "missing required output"
    names_e = {n for n, _ in expected}
    names_m = {n for n, _ in made}
    if names_e - names_m and not (names_m - names_e):
        return "missing write"
    if names_m - names_e and not (names_e - names_m):
        return "extra write"
    if names_e == names_m:
        return "wrong arguments"
    return "other"


def main() -> None:
    items = load_items()
    verdicts = load_verdicts(items)
    agg = aggregate(verdicts, items)
    wide = agg.pivot(index="state_hash", columns="judge", values="modal").join(
        items.set_index("state_hash")
    )

    show("Truth by agent", pd.crosstab(items.variant, items.truth, margins=True))

    headline = pd.DataFrame({j: summarize(agg[agg.judge == j]) for j in JUDGES}).T
    ci = [wilson(int(round(r.accuracy * r.n)), int(r.n)) for r in headline.itertuples()]
    headline["acc_ci_lo"] = [lo for lo, _ in ci]
    headline["acc_ci_hi"] = [hi for _, hi in ci]
    show("Headline, both agents, modal of five", headline.round(3))

    per_agent = pd.DataFrame(
        {
            (j, v): summarize(agg[(agg.judge == j) & (agg.variant == v)])
            for j in JUDGES
            for v in VARIANTS
        }
    ).T[["accuracy", "kappa", "pass_rate", "recall_fail"]]
    show("Per agent", per_agent.round(3))

    for j in JUDGES:
        sub = agg[agg.judge == j]
        show(f"Confusion, {j} (rows truth, columns verdict)", pd.crosstab(sub.truth, sub.modal))

    cols = JUDGES + ["truth"]
    kappas = pd.DataFrame(
        [[cohens_kappa(wide[a].tolist(), wide[b].tolist()) for b in cols] for a in cols],
        index=cols,
        columns=cols,
    )
    show("Kappa between judges and truth", kappas.round(2))

    shift = {}
    for j in JUDGES:
        for t in ["pass", "fail"]:
            rates = []
            for v in VARIANTS:
                sub = agg[(agg.judge == j) & (agg.variant == v) & (agg.truth == t)]
                rates.append(float((sub.modal == "pass").mean()))
            shift[(j, f"truth {t}")] = {"careful": rates[0], "rushed": rates[1]}
    show("Judge pass rate by agent, split by truth", pd.DataFrame(shift).T.round(2))

    repeats = {}
    for j in JUDGES:
        accs = [
            accuracy(
                verdicts[(verdicts.judge == j) & (verdicts.repeat == r)].choice.tolist(),
                verdicts[(verdicts.judge == j) & (verdicts.repeat == r)].truth.tolist(),
            )
            for r in range(5)
        ]
        sub = agg[agg.judge == j]
        repeats[j] = {
            "single_min": min(accs),
            "single_max": max(accs),
            "single_mean": float(np.mean(accs)),
            "modal": accuracy(sub.modal.tolist(), sub.truth.tolist()),
            "split_3_2": int(sub.n_pass.isin([2, 3]).sum()),
        }
    show("Single repeat versus modal", pd.DataFrame(repeats).T.round(3))

    missed = wide[(wide.truth == "fail") & (wide[MODELS] == "pass").all(axis=1)]
    tasks = {t.task_id: t for t in load_tasks()}
    kinds = Counter(miss_kind(r.variant, r.task_id, tasks) for r in missed.itertuples())
    print(
        f"\n## Failures passed by all three model judges: {len(missed)} "
        f"({missed.variant.value_counts().to_dict()}), rule-based check failed "
        f"{int((missed.code == 'fail').sum())} of them\n"
    )
    print(pd.Series(kinds).to_markdown())

    print(
        f"\n## Truncation: {int(items.truncated_results.sum())} of "
        f"{int(items.tool_results.sum())} tool results cut; "
        f"{int((items.truncated_results == 0).sum())} of {len(items)} copies untouched"
    )

    themes = {
        "missing confirmation": (
            r"without (explicit |user |the user'?s |proper |first )?(obtaining |getting )?"
            r"(explicit )?confirm|did not (obtain|get|ask for|seek|wait for|receive)[^.]{0,30}"
            r"confirm|no (explicit )?(user )?confirmation|fail(ed|ure) to (obtain|get|secure|"
            r"confirm)|lack(ed|ing|s)? (of )?(an? )?(explicit )?confirm|never (explicitly )?"
            r"confirm|skipp?(ed|ing) (the )?confirm|not (explicitly )?confirmed|unconfirmed"
        ),
        "fabricated or unsupported": (
            r"fabricat|unsupported|not supported by|hallucinat|invented|made up|inaccurate|"
            r"misstat|incorrect(ly)? (stated|informed|told|claim)"
        ),
        "policy or eligibility": (
            r"not (permitted|allowed|eligible)|violat|against (the )?policy|ineligible|"
            r"should have (refused|declined|transferred)"
        ),
        "authentication": (
            r"(without|before|prior to|did not|failed to|no)[^.]{0,20}(authenticat|verif)|"
            r"unauthenticated|unverified"
        ),
        "wrong or missing action": (
            r"wrong (item|variant|order|payment|user)|incorrect (item|variant|order|payment)|"
            r"did not (perform|complete|process|execute|make|apply)|failed to (perform|complete|"
            r"process|execute|apply)|missing (action|change)|"
            r"not (performed|completed|processed|executed)"
        ),
    }
    rows = {}
    for j in ["llm_cheap", "llm_strong"]:
        for v in VARIANTS:
            sub = verdicts[
                (verdicts.judge == j)
                & (verdicts.variant == v)
                & (verdicts.choice == "fail")
                & (verdicts.truth == "pass")
            ]
            rows[(j, v)] = {"fail_verdicts": len(sub), "conversations": sub.state_hash.nunique()}
            for name, pattern in themes.items():
                compiled = re.compile(pattern.replace("(", "(?:"), re.IGNORECASE)
                rows[(j, v)][name] = float(sub.rationale.str.contains(compiled).mean())
    show("Themes in fail rationales on runs the state check passed", pd.DataFrame(rows).T.round(2))

    cost = {}
    for j in JUDGES:
        sub = verdicts[verdicts.judge == j]
        pin, pout = PRICE_PER_MTOK[j]
        usd = (sub.input_tokens * pin + sub.output_tokens * pout) / 1e6
        cost[j] = {
            "input_tokens": sub.input_tokens.mean(),
            "output_tokens": sub.output_tokens.mean(),
            "usd_per_verdict": usd.mean(),
            "usd_per_1000": usd.mean() * 1000,
            "latency_p50_ms": sub.latency_ms.median(),
            "latency_p95_ms": sub.latency_ms.quantile(0.95),
        }
    show("Cost and latency per verdict", pd.DataFrame(cost).T.round(5))


if __name__ == "__main__":
    main()
