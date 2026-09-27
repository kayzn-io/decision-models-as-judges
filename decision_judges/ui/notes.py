"""The Technical notes panel: the full, honest account of where the ground truth came from.

Every other UI string hides the mechanism behind plain words. This module is the
single place that tells the whole story, and it is reached only two levels deep,
inside the About these conversations expander and then a collapsed Technical
notes expander. Nothing links to it. Because it is the one honest account, it is
the sole module exempt from the wording ban (see ``tests/ui/test_wording.py``).
"""

from decision_judges.ui.steps import Provenance

TECHNICAL_NOTES_TITLE = "Technical notes"

_SHORT_REF_LENGTH = 12


def _short_ref(ref: str) -> str:
    """Return the recorded version as a short code the reader can quote."""
    return ref[:_SHORT_REF_LENGTH]


def technical_notes(prov: Provenance | None) -> str:
    """Return the full provenance as a few short Markdown paragraphs.

    The account names the public source, how each conversation is produced, how
    the ground truth is decided, and the one difference between the two agents.
    The model names and the recorded version appear only when a record supplies
    them; without one the story stands on its own.
    """
    tasks = (
        "Each customer request is one of its 115 scripted tasks, acted out turn by turn "
        "by a second language model playing the customer."
    )
    if prov is not None:
        tasks += (
            f" Here the support agent is `{prov.agent_model}` "
            f"and the customer is `{prov.user_model}`."
        )
    paragraphs = [
        "The conversations come from the retail portion of tau-bench, a public test set "
        "for AI support agents published by Sierra Research.",
        tasks,
        "The support agent works against a private copy of one store's records. It can look "
        "up orders and make changes for the customer by calling tools.",
        "The ground truth for each conversation is computed by the test set's own grading "
        "code, which compares the store's records after the conversation to the expected "
        "result. No model and no person decides it.",
        "The two agents differ by a single instruction. The rushed agent is told to act "
        "without confirming details with the customer first; the careful agent is not.",
    ]
    if prov is not None:
        paragraphs.append(
            "The exact software version of the test set is recorded with every conversation, "
            f"here the short code `{_short_ref(prov.tau_bench_ref)}`."
        )
    paragraphs.append("The judges are scored against this ground truth. They are never shown it.")
    return "\n\n".join(paragraphs)
