"""Generate a tiny, deterministic Laya stub checkpoint for tests.

Running this script rebuilds the fixture under this directory: a small WordPiece
tokenizer, a tiny BERT-like encoder, and randomly initialized head weights whose
keys match :data:`decision_judges.judges.laya_model.HEAD_WEIGHT_KEYS`. The whole
checkpoint stays well under 5 MB so it can live in the repository.

Run with: ``python tests/fixtures/laya_stub/make_stub.py``.
"""

import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from decision_judges.judges.laya_model import HEAD_WEIGHT_KEYS, DecisionHead

ROOT = Path(__file__).parent
HIDDEN_SIZE = 32
SIZE_LIMIT_BYTES = 5 * 1024 * 1024
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

CORPUS = [
    "the customer asked for a refund on a damaged order",
    "the agent verified the account and issued store credit",
    "urgency is low when the ticket can wait until tomorrow",
    "urgency is high when a payment is blocking the checkout",
    "choice question about which department should own the case",
    "score question about how urgent the escalation really is",
    "noul question about whether the policy was followed exactly",
    "billing handles invoices refunds and payment disputes",
    "technical handles bugs outages and integration failures",
    "the response was polite concise and resolved the problem",
    "the state serializes the conversation into compact text",
    "options are scored at their own mask marker in the prompt",
    "the model reads the state plus a set of typed questions",
    "it returns typed answers with calibrated probabilities",
    "false means the condition does not hold in the state",
    "true means the condition clearly holds in the state",
]


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Train a small WordPiece tokenizer with the required special tokens."""
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=400, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(CORPUS, trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def dir_size_bytes(path: Path) -> int:
    """Return the total size in bytes of every file under a directory."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    """Build the stub checkpoint deterministically and assert its size."""
    for name in ("tokenizer", "encoder"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    (ROOT / "model.safetensors").unlink(missing_ok=True)

    fast_tokenizer = build_tokenizer()
    fast_tokenizer.save_pretrained(str(ROOT / "tokenizer"))

    config = BertConfig(
        vocab_size=fast_tokenizer.vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    encoder = BertModel(config)
    encoder.save_pretrained(str(ROOT / "encoder"), safe_serialization=True)

    torch.manual_seed(1)
    head = DecisionHead(HIDDEN_SIZE)
    remapped = {HEAD_WEIGHT_KEYS[key]: value for key, value in head.state_dict().items()}
    save_file(remapped, str(ROOT / "model.safetensors"))

    total = dir_size_bytes(ROOT)
    if total >= SIZE_LIMIT_BYTES:
        raise AssertionError(f"stub is {total} bytes, over the {SIZE_LIMIT_BYTES} limit")
    print(f"stub built: {total} bytes")


if __name__ == "__main__":
    main()
