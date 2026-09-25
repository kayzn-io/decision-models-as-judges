"""Run tau-bench retail tasks with a tool-calling agent under a chosen policy.

Package facts (tau-bench @ git 59a200c):

- One task runs by building an env with ``tau_bench.envs.get_env("retail",
  user_strategy="llm", user_model, task_split="test", user_provider,
  task_index)`` and solving it with
  ``tau_bench.agents.tool_calling_agent.ToolCallingAgent(tools_info=env.tools_info,
  wiki=..., model, provider, temperature).solve(env=env, task_index=idx)``.
  ``tau_bench.run.run(RunConfig)`` wraps this for whole ranges; the agent and
  env split lets a single task run with an injected policy.
- The agent system prompt is the retail policy wiki. It lives in
  ``tau_bench.envs.retail.wiki.WIKI`` (read from ``wiki.md``) and reaches the
  agent as the ``wiki`` constructor argument, which ``solve`` places as the
  ``{"role": "system"}`` message. Passing a modified string to that argument
  injects a degraded policy without editing the installed package.
- ``solve`` returns ``SolveResult(reward: float, messages: list[dict],
  info: dict, total_cost: float | None)``. Messages are OpenAI chat dicts:
  ``system``/``user``/``assistant``/``tool`` roles, assistant tool calls under
  ``message["tool_calls"]`` (each with ``id`` and ``function.name`` /
  ``function.arguments``), and tool replies as ``{"role": "tool",
  "tool_call_id", "name", "content"}``. Reward is 1.0 or 0.0; ``info`` carries
  the reward breakdown (``r_actions``/``r_outputs``) and ``source``.
- Both the agent and the user simulator call ``litellm.completion(model=...,
  custom_llm_provider=provider)``. The OpenAI-compatible base URL and key reach
  litellm through the ``OPENAI_BASE_URL``/``OPENAI_API_BASE`` and
  ``OPENAI_API_KEY`` environment variables; the study points these at one
  OpenRouter base URL with provider ``openai``.
- Token counts are not surfaced: ``SolveResult`` exposes only ``total_cost``,
  and ``info`` carries no usage. The runner therefore estimates tokens from
  message lengths at 4 characters per token for spend settlement and marks the
  estimate in the record's ``harness_info``.
"""

import json
import os
import threading
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from decision_judges.bench.load import Task
from decision_judges.config import StudyConfig
from decision_judges.progress import CancelToken, Progress, ProgressCallback, utc_now_iso
from decision_judges.spend import Reservation, Spend, SpendCapExceeded
from decision_judges.types import Usage

Variant = Literal["baseline", "degraded"]

# Pinned tau-bench commit, copied from the dependency spec in pyproject.toml
# (tau-bench @ git+https://github.com/sierra-research/tau-bench@<sha>).
TAU_BENCH_REF = "59a200c6d575d595120f1cb70fea53cef0632f6b"

# The retail policy sentence that requires explicit confirmation before any
# state-changing action. Removing it is the degraded variant. An exact match
# guards against an upstream wiki edit silently neutralizing the variant.
CONFIRMATION_RULE = (
    "- Before taking consequential actions that update the database (cancel, "
    "modify, return, exchange), you have to list the action detail and obtain "
    "explicit user confirmation (yes) to proceed."
)

# Per-task reservation estimate: a retail episode runs many tool-calling turns
# over a long policy prompt, so 60k input and 4k output tokens is a deliberate
# over-estimate that settlement later corrects to the observed usage.
EST_INPUT_TOKENS = 60_000
EST_OUTPUT_TOKENS = 4_000


class RawRunResult(BaseModel):
    """The raw outcome of one task run, before it becomes an AgentRecord."""

    reward: float
    messages: list[dict[str, object]]
    info: dict[str, object]
    est_input_tokens: int
    est_output_tokens: int


class AgentRecord(BaseModel):
    """A persisted record of one agent run under one policy variant."""

    variant: Variant
    task_id: str
    trajectory: list[dict[str, object]]
    reward: float
    harness_info: dict[str, object]
    agent_model: str
    user_model: str
    tau_bench_ref: str
    excluded: bool = False
    exclusion_reason: str | None = None


class RunSummary(BaseModel):
    """A summary of one run_variant invocation."""

    variant: Variant
    completed: int
    excluded: int
    skipped_existing: int
    pass_rate: float
    stopped_reason: str | None = None


class TaskRunner(Protocol):
    """Runs one task under a policy and reports its result and token estimate.

    Carries the models and policy it was built with so the orchestrator can
    reserve spend and stamp records without importing tau_bench.
    """

    agent_model: str
    user_model: str
    policy: str

    def __call__(self, task_index: int, policy: str) -> RawRunResult: ...


def degraded_policy(wiki_text: str) -> str:
    """Return the policy with the confirmation-before-write rule removed.

    Raise ValueError when the exact rule is absent so an upstream wiki change
    cannot silently produce a variant identical to the baseline.
    """
    if CONFIRMATION_RULE not in wiki_text:
        raise ValueError("confirmation rule not found in policy text")
    for target in (CONFIRMATION_RULE + "\n\n", "\n\n" + CONFIRMATION_RULE, CONFIRMATION_RULE):
        if target in wiki_text:
            return wiki_text.replace(target, "", 1)
    return wiki_text


def _estimate_tokens(messages: Sequence[dict[str, object]]) -> tuple[int, int]:
    """Estimate input and output tokens from message lengths at 4 chars/token."""
    chars_in = 0
    chars_out = 0
    for message in messages:
        content = message.get("content") or ""
        text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
        if message.get("role") == "assistant":
            chars_out += len(text)
        else:
            chars_in += len(text)
    return chars_in // 4, chars_out // 4


class _TauRunner:
    """The real runner: wires tau_bench with the study's models and policy."""

    def __init__(self, study: StudyConfig, variant: Variant) -> None:
        from tau_bench.envs.retail.wiki import WIKI

        self.agent_model = study.models.agent
        self.user_model = study.models.user_sim
        self._provider = "openai"
        self._temperature = 0.0
        self._base_url = study.llm_base_url
        self.policy = WIKI if variant == "baseline" else degraded_policy(WIKI)
        os.environ["OPENAI_BASE_URL"] = self._base_url
        os.environ["OPENAI_API_BASE"] = self._base_url

    def __call__(self, task_index: int, policy: str) -> RawRunResult:
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        from tau_bench.envs import get_env

        env = get_env(
            "retail",
            user_strategy="llm",
            user_model=self.user_model,
            task_split="test",
            user_provider=self._provider,
            task_index=task_index,
        )
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=policy,
            model=self.agent_model,
            provider=self._provider,
            temperature=self._temperature,
        )
        result = agent.solve(env=env, task_index=task_index)
        est_input, est_output = _estimate_tokens(result.messages)
        info = dict(result.info)
        info["token_estimate"] = "chars_over_4"
        return RawRunResult(
            reward=result.reward,
            messages=result.messages,
            info=info,
            est_input_tokens=est_input,
            est_output_tokens=est_output,
        )


def build_tau_runner(study: StudyConfig, variant: Variant) -> TaskRunner:
    """Build a runner bound to the study's models, base URL, and variant policy."""
    return _TauRunner(study, variant)


def _record_path(out_dir: Path, variant: Variant, task_id: str) -> Path:
    """Return the JSON path for one task's record."""
    return out_dir / variant / f"{task_id}.json"


def _load_existing(path: Path) -> AgentRecord | None:
    """Return the record already at path if it validates, else None."""
    if not path.is_file():
        return None
    try:
        return AgentRecord.model_validate_json(path.read_text())
    except (ValueError, OSError):
        return None


def _write_record(path: Path, record: AgentRecord) -> None:
    """Write a record atomically as stable sorted JSON via temp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.model_dump(mode="json"), sort_keys=True, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


def _append_exclusion(out_dir: Path, variant: Variant, task_id: str, reason: str) -> None:
    """Append one exclusion line to the run's exclusions log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exclusions.md", "a") as handle:
        handle.write(f"- {variant}/{task_id}: {reason}\n")


def _task_index(task_id: str) -> int:
    """Recover the positional tau-bench index from a ``retail-{index}`` id."""
    return int(task_id.rsplit("-", 1)[1])


def _completed_record(
    variant: Variant, task: Task, runner: TaskRunner, raw: RawRunResult
) -> AgentRecord:
    """Build a record for a task that finished."""
    return AgentRecord(
        variant=variant,
        task_id=task.task_id,
        trajectory=list(raw.messages),
        reward=raw.reward,
        harness_info=dict(raw.info),
        agent_model=runner.agent_model,
        user_model=runner.user_model,
        tau_bench_ref=TAU_BENCH_REF,
    )


def _excluded_record(
    variant: Variant, task: Task, runner: TaskRunner, exc: Exception
) -> AgentRecord:
    """Build a record for a task whose run raised."""
    return AgentRecord(
        variant=variant,
        task_id=task.task_id,
        trajectory=[],
        reward=0.0,
        harness_info={},
        agent_model=runner.agent_model,
        user_model=runner.user_model,
        tau_bench_ref=TAU_BENCH_REF,
        excluded=True,
        exclusion_reason=f"{type(exc).__name__}: {exc}",
    )


def run_variant(
    variant: Variant,
    tasks: list[Task],
    runner: TaskRunner,
    spend: Spend,
    out_dir: Path,
    *,
    concurrency: int,
    stage: str = "agent",
    on_progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> RunSummary:
    """Run every not-yet-recorded task under one policy variant, resumably.

    Reserve spend before each run, run tasks concurrently, settle with the
    runner's reported token estimate, and write each record atomically. A
    runner exception excludes that task and logs it; a spend cap stops
    scheduling new tasks while in-flight ones finish. When ``on_progress`` is
    given, a snapshot is reported as each task completes; when ``cancel`` is
    set, no new tasks are scheduled, in-flight tasks finish, and the summary
    records a ``cancelled`` stop.
    """
    (out_dir / variant).mkdir(parents=True, exist_ok=True)

    pending: list[Task] = []
    skipped = 0
    for task in tasks:
        if _load_existing(_record_path(out_dir, variant, task.task_id)) is not None:
            skipped += 1
        else:
            pending.append(task)

    lock = threading.Lock()

    def _run_task(task: Task, reservation: Reservation) -> AgentRecord:
        path = _record_path(out_dir, variant, task.task_id)
        try:
            raw = runner(_task_index(task.task_id), runner.policy)
        except Exception as exc:  # noqa: BLE001 - any failure excludes one task
            with lock:
                spend.cancel(reservation)
                _append_exclusion(out_dir, variant, task.task_id, f"{type(exc).__name__}: {exc}")
            record = _excluded_record(variant, task, runner, exc)
            _write_record(path, record)
            return record
        with lock:
            spend.settle(
                reservation,
                Usage(input_tokens=raw.est_input_tokens, output_tokens=raw.est_output_tokens),
            )
        record = _completed_record(variant, task, runner, raw)
        _write_record(path, record)
        return record

    total = len(pending)
    started_at = utc_now_iso()
    stopped_reason: str | None = None
    records: list[AgentRecord] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        queue = list(pending)
        index = 0
        in_flight: set[Future[AgentRecord]] = set()

        def submit_more() -> None:
            nonlocal index, stopped_reason
            while len(in_flight) < concurrency and index < len(queue):
                if cancel is not None and cancel.is_cancelled:
                    stopped_reason = stopped_reason or "cancelled"
                    return
                task = queue[index]
                try:
                    with lock:
                        reservation = spend.reserve(
                            stage, runner.agent_model, EST_INPUT_TOKENS, EST_OUTPUT_TOKENS
                        )
                except SpendCapExceeded as exc:
                    stopped_reason = stopped_reason or str(exc)
                    return
                index += 1
                in_flight.add(executor.submit(_run_task, task, reservation))

        submit_more()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.discard(future)
                record = future.result()
                records.append(record)
                if on_progress is not None:
                    if record.excluded:
                        label = "excluded"
                    else:
                        label = "pass" if record.reward >= 1.0 else "fail"
                    on_progress(
                        Progress(
                            step_id=stage,
                            done=len(records),
                            total=total,
                            spent_usd=spend.spent(stage),
                            cap_usd=spend.cap(stage),
                            last_item=f"{record.task_id} · {label}",
                            started_at=started_at,
                        )
                    )
            submit_more()

    completed = sum(1 for record in records if not record.excluded)
    excluded = sum(1 for record in records if record.excluded)
    rewards = [record.reward for record in records if not record.excluded]
    pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
    return RunSummary(
        variant=variant,
        completed=completed,
        excluded=excluded,
        skipped_existing=skipped,
        pass_rate=pass_rate,
        stopped_reason=stopped_reason,
    )
