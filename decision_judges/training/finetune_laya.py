"""Fine-tune the Laya decision model against strictly proper scoring rewards.

The loop reuses the vendored training pieces: it encodes each labelled example
into per-option marker sequences, runs the decision model, and maximizes the
proper-scoring reward of the softmax distribution against one-hot targets with
AdamW. It saves a checkpoint that reloads strictly through
:class:`LayaDecisionModel`, fits per-bucket calibration temperatures on held-out
data, and drives k-fold cross validation with one manifest per fold.
"""

import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from decision_judges.judges.laya_model import LayaDecisionModel
from decision_judges.judges.laya_vendor import rl_common
from decision_judges.metrics import ece
from decision_judges.serialize import StateProfile, StateRecord
from decision_judges.training.folds import assign_folds, train_test_split
from decision_judges.types import Question, QuestionKind

# Calibration temperatures are searched on this grid and clamped to [0.5, 5.0].
TEMP_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


class TrainingExample(BaseModel):
    """One labelled state: its text, the questions asked, and the true labels."""

    state: str
    questions: list[Question]
    labels: dict[str, int]


class Manifest(BaseModel):
    """Provenance and calibration for one fold's fine-tuned checkpoint."""

    base_checkpoint_hash: str
    fold: int
    k: int
    seed: int
    epochs: int
    learning_rate: float
    batch_size: int
    train_task_ids: list[str]
    test_task_ids: list[str]
    temperature_by_options: dict[str, float]
    n_examples: int
    created_at: datetime


def _label_for(question: Question, reward: float) -> int:
    """Return the true label index for a question given a run's reward.

    A pass (reward at or above one) is the first choice option and the true
    slot of a noul question; a fail is the second choice option and the false
    slot.
    """
    passed = reward >= 1.0
    if question.kind is QuestionKind.noul:
        return 1 if passed else 0
    if question.kind is QuestionKind.choice:
        return 0 if passed else 1
    raise ValueError(f"cannot derive an outcome label for a {question.kind} question")


def build_training_examples(
    states: Mapping[str, StateRecord],
    rewards: Mapping[str, float],
    questions: Sequence[Question],
) -> list[TrainingExample]:
    """Pair each compact state with outcome labels derived from its reward.

    Raise ``ValueError`` if any state was serialized under a non-compact
    profile, since the model reads only the compact view.
    """
    question_list = list(questions)
    examples: list[TrainingExample] = []
    for task_id, state in states.items():
        if state.profile is not StateProfile.compact:
            raise ValueError(f"{task_id} has profile {state.profile.value}, expected compact")
        reward = rewards[task_id]
        labels = {question.id: _label_for(question, reward) for question in question_list}
        examples.append(TrainingExample(state=state.text, questions=question_list, labels=labels))
    return examples


def to_vendor_record(example: TrainingExample) -> dict[str, object]:
    """Render an example into the vendored ``{state, qs:[{t, ins, crit, y}]}`` record.

    Choice criteria are option keys with empty descriptions, score criteria are
    the ordered level descriptions, and noul questions carry no criteria.
    """
    questions: list[dict[str, object]] = []
    for question in example.questions:
        label = example.labels[question.id]
        if question.kind is QuestionKind.choice:
            crit = {option: "" for option in question.options or []}
            questions.append({"t": "choice", "ins": question.text, "crit": crit, "y": label})
        elif question.kind is QuestionKind.score:
            questions.append(
                {
                    "t": "score",
                    "ins": question.text,
                    "crit": list(question.levels or []),
                    "y": label,
                }
            )
        else:
            questions.append({"t": "noul", "ins": question.text, "y": label})
    return {"state": example.state, "qs": questions}


def write_manifest(directory: Path, manifest: Manifest) -> None:
    """Write a manifest as ``manifest.json`` inside a checkpoint directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def read_manifest(directory: Path) -> Manifest:
    """Read the ``manifest.json`` from a checkpoint directory."""
    return Manifest.model_validate_json((Path(directory) / "manifest.json").read_text())


def _load_cfg(directory: Path) -> dict[str, object]:
    """Read the vendored agent config from a checkpoint directory."""
    return json.loads((Path(directory) / "rl_agent_config.json").read_text())


def _encode(
    examples: Sequence[TrainingExample],
    tokenizer: object,
    cfg: dict[str, object],
    rng: random.Random | None,
    *,
    train: bool,
) -> list[list[dict[str, object]]]:
    """Encode each example into a group of per-question marker sequences."""
    groups: list[list[dict[str, object]]] = []
    for example in examples:
        items = rl_common.encode_record(to_vendor_record(example), tokenizer, cfg, rng, train)
        if items:
            groups.append(items)
    return groups


def _batches(
    groups: Sequence[list[dict[str, object]]], batch_size: int
) -> list[list[list[dict[str, object]]]]:
    """Split encoded groups into batches of at most ``batch_size`` examples."""
    return [list(groups[start : start + batch_size]) for start in range(0, len(groups), batch_size)]


def _step(
    model: object,
    optimizer: object,
    batch: list[list[dict[str, object]]],
    pad_id: int,
    device: object,
) -> float:
    """Run one optimization step maximizing the proper-scoring reward."""
    import torch

    collated = rl_common.collate_items(batch, pad_id)
    optimizer.zero_grad()  # type: ignore[attr-defined]
    logits, _ = model(  # type: ignore[operator]
        collated["input_ids"].to(device),
        collated["attention_mask"].to(device),
        collated["marker_pos"].to(device),
        collated["marker_mask"].to(device),
        collated["qtype"].to(device),
    )
    probs = torch.softmax(logits, dim=-1)
    reward = rl_common.proper_reward(
        probs,
        collated["target"].to(device),
        collated["qtype"].to(device),
        collated["marker_mask"].to(device),
    )
    loss = -reward.mean()
    loss.backward()
    optimizer.step()  # type: ignore[attr-defined]
    return float(loss.detach())


def _save_checkpoint(model: object, base_dir: Path, out_dir: Path) -> None:
    """Save weights and copy config, encoder config, and tokenizer for strict reload."""
    from safetensors.torch import save_file

    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()  # type: ignore[attr-defined]
    }
    save_file(state, str(out_dir / "model.safetensors"))
    shutil.copy2(base_dir / "rl_agent_config.json", out_dir / "rl_agent_config.json")
    (out_dir / "encoder").mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_dir / "encoder" / "config.json", out_dir / "encoder" / "config.json")
    tokenizer_out = out_dir / "tokenizer"
    if tokenizer_out.exists():
        shutil.rmtree(tokenizer_out)
    shutil.copytree(base_dir / "tokenizer", tokenizer_out)


def fine_tune(
    base_dir: Path,
    examples: Sequence[TrainingExample],
    out_dir: Path,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: str = "cpu",
    max_steps: int | None = None,
) -> Path:
    """Fine-tune the decision model on labelled examples and save a checkpoint.

    Loads the base checkpoint, encodes each example, and trains encoder and head
    with AdamW to maximize the proper-scoring reward of the masked softmax
    against one-hot targets. The saved directory reloads strictly through
    :meth:`LayaDecisionModel.from_pretrained`.
    """
    import torch
    from torch.optim import AdamW

    torch.manual_seed(seed)
    base = LayaDecisionModel.from_pretrained(str(base_dir), device=device)
    model = base.model
    tokenizer = base.tokenizer
    cfg = _load_cfg(base_dir)
    pad_id = tokenizer.pad_token_id or 0
    rng = random.Random(seed)
    groups = _encode(examples, tokenizer, cfg, rng, train=True)

    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    torch_device = torch.device(device)
    steps = 0
    for _ in range(epochs):
        for batch in _batches(groups, batch_size):
            _step(model, optimizer, batch, pad_id, torch_device)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break

    model.eval()
    _save_checkpoint(model, base_dir, out_dir)
    return Path(out_dir)


def _softmax(values: Sequence[float]) -> list[float]:
    """Return a numerically stable softmax over a list of logits."""
    highest = max(values)
    exps = [math.exp(value - highest) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _fit_bucket_temperature(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Return the grid temperature minimizing ECE of max-probability correctness."""
    best_temperature = 1.0
    best_ece = float("inf")
    for temperature in TEMP_GRID:
        confidences: list[float] = []
        correct: list[float] = []
        for row, label in zip(rows, labels, strict=True):
            probs = _softmax([value / temperature for value in row])
            prediction = max(range(len(probs)), key=lambda index: probs[index])
            confidences.append(max(probs))
            correct.append(1.0 if prediction == label else 0.0)
        current = ece(confidences, correct)
        if current < best_ece:
            best_ece = current
            best_temperature = float(temperature)
    return best_temperature


def fit_temperatures(model_dir: Path, examples: Sequence[TrainingExample]) -> dict[str, float]:
    """Fit one calibration temperature per ``(type, option-count)`` bucket.

    Runs the checkpoint over the held-out examples, groups the option logits by
    :func:`rl_common.temp_bucket`, and grid-searches the temperature that
    minimizes the expected calibration error of max-probability correctness.
    """
    import torch

    loaded = LayaDecisionModel.from_pretrained(str(model_dir))
    model = loaded.model
    tokenizer = loaded.tokenizer
    cfg = _load_cfg(model_dir)
    groups = _encode(examples, tokenizer, cfg, None, train=False)
    if not groups:
        return {}
    pad_id = tokenizer.pad_token_id or 0
    collated = rl_common.collate_items(groups, pad_id)
    with torch.no_grad():
        logits, _ = model(
            collated["input_ids"],
            collated["attention_mask"],
            collated["marker_pos"],
            collated["marker_mask"],
            collated["qtype"],
        )

    rows: dict[str, list[list[float]]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    for index in range(len(collated["meta"])):
        label = int(collated["label"][index])
        if label < 0:
            continue
        count = int(collated["marker_mask"][index].sum())
        qtype = int(collated["qtype"][index])
        bucket = rl_common.temp_bucket(qtype, count)
        rows[bucket].append([float(value) for value in logits[index, :count].tolist()])
        labels[bucket].append(label)

    return {bucket: _fit_bucket_temperature(rows[bucket], labels[bucket]) for bucket in rows}


def _checkpoint_hash(directory: Path) -> str:
    """Return the sha256 of a checkpoint's ``model.safetensors``."""
    return hashlib.sha256((Path(directory) / "model.safetensors").read_bytes()).hexdigest()


def run_cross_validation(
    base_dir: Path,
    states: Mapping[str, StateRecord],
    rewards: Mapping[str, float],
    questions: Sequence[Question],
    out_root: Path,
    *,
    k: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    max_steps: int | None = None,
) -> list[Manifest]:
    """Fine-tune one checkpoint per fold, calibrating on that fold's held-out ids.

    Each fold trains on the other folds and fits temperatures on its own test
    ids, so a task never calibrates a model that trained on it. Writes one
    ``laya-g3-fold{f}`` directory with weights and a manifest per fold.
    """
    base_dir = Path(base_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    base_hash = _checkpoint_hash(base_dir)
    assignments = assign_folds(list(states.keys()), k, seed)
    question_list = list(questions)

    manifests: list[Manifest] = []
    for fold in range(k):
        train_ids, test_ids = train_test_split(assignments, fold)
        train_examples = build_training_examples(
            {task_id: states[task_id] for task_id in train_ids}, rewards, question_list
        )
        test_examples = build_training_examples(
            {task_id: states[task_id] for task_id in test_ids}, rewards, question_list
        )
        out_dir = out_root / f"laya-g3-fold{fold}"
        fine_tune(
            base_dir,
            train_examples,
            out_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=seed,
            device=device,
            max_steps=max_steps,
        )
        temperatures = fit_temperatures(out_dir, test_examples)
        manifest = Manifest(
            base_checkpoint_hash=base_hash,
            fold=fold,
            k=k,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            train_task_ids=train_ids,
            test_task_ids=test_ids,
            temperature_by_options=temperatures,
            n_examples=len(train_examples),
            created_at=datetime.now(UTC),
        )
        write_manifest(out_dir, manifest)
        manifests.append(manifest)
    return manifests


def checkpoint_for_task(out_root: Path, task_id: str) -> Path:
    """Return the fold directory whose manifest lists a task as a test id."""
    out_root = Path(out_root)
    for directory in sorted(path for path in out_root.iterdir() if path.is_dir()):
        if not (directory / "manifest.json").is_file():
            continue
        if task_id in read_manifest(directory).test_task_ids:
            return directory
    raise KeyError(f"no fold checkpoint holds {task_id!r} in its test set")
