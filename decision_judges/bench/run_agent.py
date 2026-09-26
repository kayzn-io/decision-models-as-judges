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
import re
import threading
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
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


class MissingCredentials(RuntimeError):
    """Raised before any task runs when the API key is missing or empty."""


_CONFIG_MARKERS = ("credentials", "api key", "authentication", "missing credentials")
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "internalservererror",
    "serviceunavailable",
    "ratelimit",
    "rate limit",
)


def _classify_text(text: str) -> Literal["config", "transient", "task"]:
    """Classify failure text: config first, then transient, otherwise task."""
    lowered = text.lower()
    if any(marker in lowered for marker in _CONFIG_MARKERS) or re.search(r"\b(401|403)\b", lowered):
        return "config"
    if re.search(r"\b(429|5\d\d)\b", lowered) or any(
        marker in lowered for marker in _TRANSIENT_MARKERS
    ):
        return "transient"
    return "task"


def classify_failure(exc: BaseException) -> Literal["config", "transient", "task"]:
    """Return the class of a run failure.

    'config' marks a missing or bad credential or authorization failure, which a
    rerun with a working key fixes. 'transient' marks a timeout, dropped
    connection, rate limit, or server error, which a later rerun may clear.
    'task' marks a genuine task-level failure such as an environment exception.
    Only 'task' failures are recorded as exclusions; the rest leave the task
    pending for the next run.
    """
    return _classify_text(f"{type(exc).__name__}: {exc}")


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
    failed_transient: int = 0
    failed_config: int = 0


class TaskRunner(Protocol):
    """Runs one task under a policy and reports its result and token estimate.

    Carries the models and policy it was built with so the orchestrator can
    reserve spend and stamp records without importing tau_bench.
    """

    agent_model: str
    user_model: str
    policy: str

    def __call__(self, task_index: int, policy: str) -> RawRunResult: ...


DEGRADED_RULE = (
    "- Act on the user's request as soon as you have the information needed. Do not list the "
    "action details back to the user and do not wait for confirmation before cancelling, "
    "modifying, returning, or exchanging."
)
"""The instruction that replaces the confirmation rule in the degraded variant.

Merely deleting the confirmation rule does not change the agent's behaviour:
the model confirms by habit and the two variants end up statistically
identical. Telling it to act at once on the first version of the request does
change behaviour, because customers in these conversations often revise the
request during the confirmation exchange, and acting early produces the wrong
final state, which the grader catches.
"""


def degraded_policy(wiki_text: str) -> str:
    """Return the policy with the confirmation rule replaced by an act-at-once rule.

    Raise ValueError when the exact confirmation rule is absent so an upstream
    wiki change cannot silently produce a variant identical to the baseline.
    """
    if CONFIRMATION_RULE not in wiki_text:
        raise ValueError("confirmation rule not found in policy text")
    return wiki_text.replace(CONFIRMATION_RULE, DEGRADED_RULE, 1)


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
        self._key_env = "OPENROUTER_API_KEY"
        self.policy = WIKI if variant == "baseline" else degraded_policy(WIKI)
        os.environ["OPENAI_BASE_URL"] = self._base_url
        os.environ["OPENAI_API_BASE"] = self._base_url

    def preflight(self) -> None:
        """Fail before any task when the API key is missing, and hand it to litellm.

        tau_bench calls litellm with the ``openai`` provider, which reads
        ``OPENAI_API_KEY``. The study holds one OpenRouter key, so the check and
        the hand-off happen together: a missing key raises before any task, and a
        present one is copied into the variable litellm reads.
        """
        key = os.environ.get(self._key_env)
        if not key:
            raise MissingCredentials(
                f"{self._key_env} is not set. Set it in the app sidebar or run "
                f"export {self._key_env}=... before this step."
            )
        os.environ["OPENAI_API_KEY"] = key

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
    """Return the record at path when it counts as done, else None.

    A completed record (``excluded is False``) always counts. An excluded record
    counts only when its stored reason classifies as a 'task' failure, a genuine
    task-level exclusion. A record excluded for a 'config' or 'transient' reason
    does not count, so a rerun with a working key or a cleared outage runs that
    task again instead of treating the earlier failure as permanent.
    """
    if not path.is_file():
        return None
    try:
        record = AgentRecord.model_validate_json(path.read_text())
    except (ValueError, OSError):
        return None
    if not record.excluded:
        return record
    reason = record.exclusion_reason or ""
    if reason and _classify_text(reason) == "task":
        return record
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


@dataclass
class _RunOutcome:
    """The outcome of one task: its written record, if any, and failure class."""

    task_id: str
    record: AgentRecord | None
    failure_class: Literal["config", "transient", "task"] | None
    message: str
    reward: float


def _progress_label(outcome: _RunOutcome) -> str:
    """Return the short progress label for one task outcome."""
    if outcome.failure_class == "task":
        return "excluded"
    if outcome.failure_class is not None:
        return outcome.failure_class
    return "pass" if outcome.reward >= 1.0 else "fail"


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

    Before any task is submitted, an optional ``preflight`` hook on the runner
    may raise :class:`MissingCredentials` to stop the run before it spends
    anything. Reserve spend before each run, run tasks concurrently, settle with
    the runner's reported token estimate, and write each record atomically.

    A task-level failure excludes that task and logs it. A 'config' or
    'transient' failure is not written to disk: it is counted in the summary and
    leaves the task pending for the next run. When the first three completed
    tasks all fail for a 'config' or 'transient' reason, scheduling stops and the
    summary records an aborted stop so a broken setup ends in seconds. A spend
    cap stops scheduling new tasks while in-flight ones finish; a cancel token
    does the same and records a cancelled stop.
    """
    (out_dir / variant).mkdir(parents=True, exist_ok=True)

    pending: list[Task] = []
    skipped = 0
    for task in tasks:
        if _load_existing(_record_path(out_dir, variant, task.task_id)) is not None:
            skipped += 1
        else:
            pending.append(task)

    preflight = getattr(runner, "preflight", None)
    if pending and preflight is not None:
        preflight()

    lock = threading.Lock()

    def _run_task(task: Task, reservation: Reservation) -> _RunOutcome:
        path = _record_path(out_dir, variant, task.task_id)
        try:
            raw = runner(_task_index(task.task_id), runner.policy)
        except Exception as exc:  # noqa: BLE001 - classified, not always excluded
            failure = classify_failure(exc)
            message = f"{type(exc).__name__}: {exc}"
            with lock:
                spend.cancel(reservation)
            if failure != "task":
                return _RunOutcome(task.task_id, None, failure, message, 0.0)
            with lock:
                _append_exclusion(out_dir, variant, task.task_id, message)
            record = _excluded_record(variant, task, runner, exc)
            _write_record(path, record)
            return _RunOutcome(task.task_id, record, "task", message, 0.0)
        with lock:
            spend.settle(
                reservation,
                Usage(input_tokens=raw.est_input_tokens, output_tokens=raw.est_output_tokens),
            )
        record = _completed_record(variant, task, runner, raw)
        _write_record(path, record)
        return _RunOutcome(task.task_id, record, None, "", raw.reward)

    total = len(pending)
    started_at = utc_now_iso()
    stopped_reason: str | None = None
    outcomes: list[_RunOutcome] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        queue = list(pending)
        index = 0
        aborted = False
        in_flight: set[Future[_RunOutcome]] = set()

        def submit_more() -> None:
            nonlocal index, stopped_reason
            if aborted:
                return
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
                outcome = future.result()
                outcomes.append(outcome)
                if (
                    not aborted
                    and len(outcomes) >= 3
                    and all(item.failure_class in ("config", "transient") for item in outcomes[:3])
                ):
                    aborted = True
                    stopped_reason = f"aborted: the first 3 tasks failed with {outcomes[0].message}"
                if on_progress is not None:
                    on_progress(
                        Progress(
                            step_id=stage,
                            done=len(outcomes),
                            total=total,
                            spent_usd=spend.spent(stage),
                            cap_usd=spend.cap(stage),
                            last_item=f"{outcome.task_id} · {_progress_label(outcome)}",
                            started_at=started_at,
                        )
                    )
            submit_more()

    completed = sum(1 for o in outcomes if o.record is not None and not o.record.excluded)
    excluded = sum(1 for o in outcomes if o.record is not None and o.record.excluded)
    failed_config = sum(1 for o in outcomes if o.failure_class == "config")
    failed_transient = sum(1 for o in outcomes if o.failure_class == "transient")
    rewards = [o.reward for o in outcomes if o.record is not None and not o.record.excluded]
    pass_rate = sum(rewards) / len(rewards) if rewards else 0.0
    return RunSummary(
        variant=variant,
        completed=completed,
        excluded=excluded,
        skipped_existing=skipped,
        pass_rate=pass_rate,
        stopped_reason=stopped_reason,
        failed_transient=failed_transient,
        failed_config=failed_config,
    )
