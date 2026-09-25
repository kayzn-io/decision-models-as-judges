# version: 1

You read one retail customer-service conversation that failed to complete the
user's request under the policy. Pick the single failure type that best explains
why it failed. Answer in a neutral tone and do not assume facts the transcript
does not state.

- wrong_or_missing_action: a required action was skipped or the wrong action was taken.
- unrequested_write: the agent changed an order or record the user did not ask to change.
- skipped_confirmation: an order-changing write was made without the user's explicit confirmation.
- identity_not_verified: the agent acted before confirming the user's identity as the policy requires.
- wrong_arguments: the right action was called with incorrect arguments, such as the wrong order, item, or amount.
- premature_end: the conversation ended before the request was resolved.
- policy_misapplied: the agent misread the policy, refusing a permitted request or allowing a forbidden one.
- other: the failure does not fit any category above.
