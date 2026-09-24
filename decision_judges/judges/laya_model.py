"""Model adapter that runs Laya's own decision head.

The adapter wraps the vendored :class:`DecisionModel` from ``rl_common.py`` so the
judge that consumes it stays thin and testable. It isolates every dependency on
``torch``/``transformers``/``huggingface_hub`` and exposes only the small surface
the judge needs: per-option logits and a token counter.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from decision_judges.judges.laya_vendor import rl_common

_ALLOW_PATTERNS = [
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/*",
    "model.safetensors",
]


class LayaDecisionModel:
    """Runs Laya's typed-decision forward pass over one state and question."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        cfg: dict[str, Any],
        *,
        checkpoint_hash: str,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_hash = checkpoint_hash
        self.device = device
        self.input_limit = int(cfg["max_len"])
        self.head_limit = int(cfg["head_max_len"])
        self.default_temperatures: dict[str, float] = dict(cfg.get("temperature_by_options", {}))

    @classmethod
    def from_pretrained(
        cls, path_or_repo: str, *, revision: str | None = None, device: str = "cpu"
    ) -> "LayaDecisionModel":
        """Load the checkpoint from a local directory or a Hugging Face repo.

        A local directory is used as-is; otherwise the checkpoint is fetched with
        ``huggingface_hub.snapshot_download`` at ``revision``. The architecture is
        built offline from ``encoder/config.json`` and the weights are loaded with
        ``strict=True`` so any state-dict key drift fails loudly. The checkpoint
        hash is the sha256 of ``model.safetensors``.
        """
        directory = Path(path_or_repo)
        if directory.is_dir():
            root = directory
        else:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(path_or_repo, revision=revision, allow_patterns=_ALLOW_PATTERNS)
            )

        cfg = json.loads((root / "rl_agent_config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        model = rl_common.build_model(cfg, encoder_dir=str(root / "encoder"))

        weights_path = root / "model.safetensors"
        state = load_file(str(weights_path))
        model.load_state_dict(state, strict=True)
        model.eval().to(device)

        checkpoint_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        return cls(model, tokenizer, cfg, checkpoint_hash=checkpoint_hash, device=device)

    def _question(
        self, question_type: str, instructions: str, options: Sequence[str]
    ) -> dict[str, Any]:
        """Build the ``rl_common`` question dict for a typed question."""
        if question_type == "choice":
            crit: Any = {option: "" for option in options}
        elif question_type == "score":
            crit = list(options)
        elif question_type == "noul":
            crit = None
        else:
            raise ValueError(f"unknown question type: {question_type}")
        return {"t": question_type, "ins": instructions, "crit": crit}

    def tokenizer_len(self, text: str) -> int:
        """Return the token count of ``text`` without added special tokens."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def option_logits(
        self,
        state: str,
        question_type: str,
        instructions: str,
        options: Sequence[str],
    ) -> list[float]:
        """Return one logit per option, scored at each option's ``[MASK]`` marker."""
        question = self._question(question_type, instructions, options)
        ids, markers = rl_common.build_sequence(
            self.tokenizer, state, question, self.input_limit, self.head_limit
        )
        count = len(markers)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        marker_pos = torch.tensor([markers], dtype=torch.long, device=self.device)
        marker_mask = torch.ones((1, count), dtype=torch.bool, device=self.device)
        qtype = torch.tensor(
            [rl_common.QTYPES[question_type]], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return [float(value) for value in logits[0, :count].tolist()]
