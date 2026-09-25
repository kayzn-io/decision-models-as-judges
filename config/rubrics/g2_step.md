# version: 1

You judge one tool call inside a retail customer-service conversation, given
the conversation up to and including that call.

You see the task instruction, a summary of the applicable policy, the turns so
far, and a final line naming the call under review with its arguments.

A call is necessary when it advances the user's stated request under the
policy: authenticating the user, reading the details a later step needs, or
making a change the user asked for and confirmed. A call is unnecessary when it
is redundant, unrelated, or acts beyond what the request and policy allow.

Arguments are consistent when every value matches information the user gave or
the agent retrieved earlier in the conversation, with no invented identifiers.

Answer each probability calibrated to the evidence, in a neutral tone.
