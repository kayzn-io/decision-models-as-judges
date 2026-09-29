# Choosing a judge: what 4,600 verdicts say

An analysis of the four judges that read the full conversation text, written
for someone deciding what kind of judge to put in an agent evaluation loop.
The local model Laya and the short reading copy made for it are left out: Laya
judged only 69 conversations before its run was interrupted, and the short
copy exists only to fit Laya's 512-token limit, so neither adds to a comparison
of judges that can read a whole conversation.

Analysis date: 2026-09-29. Every number below comes from the committed caches
and can be regenerated without a paid call; see
[Reproducing the numbers](#reproducing-the-numbers). An earlier version of this
document (2026-09-28) analyzed the same judges on reading copies whose tool
results had been cut at 600 characters. That cut turned out to be a mistake
that changed the results; Finding 5 describes it and reports both before and
after. Everything else here is on the uncut copies.

## Summary

1. Against tau-bench's outcome ground truth, the model judges are a little
   better than guessing. gpt-5 agrees with the truth on 60% of conversations,
   Jev and gpt-4o-mini on 57%, on a task where always answering pass scores
   49%. gpt-5's margin over always-pass is 11 points (95% CI 3.5 to 19); Jev's
   and gpt-4o-mini's are 7 points each (CI 0.4 to 14 and 3 to 12). The
   rule-based check, which reads the task's answer key, scores 79%.

2. Accuracy hides how differently the three behave. gpt-4o-mini passes 88% of
   everything and catches fewer than one failure in five. Jev passes 72% and
   catches one in three, but its scores rank a real pass above a real fail 73%
   of the time, the best of the three, and it gives the same verdict on all
   five repeats for 99% of conversations. gpt-5 is the most balanced (64% pass,
   catches nearly half of failures) and, at 80% unanimous, the least
   repeatable.

3. Jev and gpt-5 are the same kind of judge. They agree with each other at
   kappa 0.50 and with the truth at 0.14 and 0.21; each agrees with gpt-4o-mini
   at under 0.10. On ranking they tie (AUROC 0.73 against 0.69, difference
   within noise). Both react to process: when the rushed agent stops confirming
   before it acts, Jev's pass rate falls 37 points and gpt-5's 17, against a
   4-point fall in the truth.

4. The failures every reading judge misses are detail errors. Forty-four
   failed conversations were passed by all three model judges. In 21 the agent
   made the right kind of change with the wrong item, variant, or payment
   method; in 8 it dropped one of several requested changes; in 5 it omitted a
   required figure. The rule-based check caught 40 of the 44.

5. Cutting tool output was our mistake and it changed the results. The first
   version of the reading copy cut every tool result at 600 characters, so
   judges were asked to check the agent's statements against evidence that had
   been removed. gpt-5 failed correct runs for "unsupported facts" that were in
   the raw tool output. With the copies whole, gpt-5's AUROC rose from 0.58 to
   0.69, Jev's from 0.65 to 0.73, and gpt-4o-mini's did not move. The per-judge
   ranking changed: gpt-5 went from the least repeatable judge at chance
   accuracy to the most accurate.

6. No judge's stated confidence is usable as a probability. Every model
   judge's Brier score on its pass probability is at or above 0.27, the score
   of a coin being 0.25. Correction on held-out conversations fixes the gap but
   not the ranking. A cascade gated on gpt-4o-mini's confidence reached 62% at
   $0.008 per conversation, the best point on the cost curve, but that is one
   or two points over either judge alone.

7. Jev and gpt-5 tie on quality at a 150× price difference. Per thousand
   verdicts on the uncut copies: Jev $0.19 at 0.17 s median, gpt-4o-mini
   $0.65 at 2.1 s, gpt-5 $28.01 at 28 s. gpt-5 spent 97% of the full-copy
   judging budget.

8. For someone choosing: if you can write down the expected end state, check
   it in code. If you cannot, a reading judge will audit process, not outcome.
   Jev does that audit as well as gpt-5 at a small fraction of the cost, with a
   reproducible verdict from one call; gpt-5 adds a written rationale and the
   best default pass/fail line; gpt-4o-mini does not do the audit at all.
   Whatever you choose, give it the whole tool output, calibrate its threshold
   on conversations you have labelled, and do not act on its probabilities raw.

## What was measured

### The judges

| Judge | What it is | What it sees | How it answers | Repeatable |
| --- | --- | --- | --- | --- |
| Rule-based check | Code that compares the agent's tool calls and final message to the task's expected actions and required outputs | The raw trajectory and the task's answer key | pass/fail at confidence 1.0 | Always |
| Jev | `jev-1.13`, a hosted typed decision model, served through OpenRouter | The reading copy | Probabilities for the two typed questions; no free text | 99% of conversations get five identical verdicts |
| Fast text model | `openai/gpt-4o-mini` | The reading copy and the rubric | A written rationale plus a structured pass/fail and confidence | 96% |
| Strong text model | `openai/gpt-5` | The reading copy and the rubric | A written rationale (about 2,300 output tokens) plus structured pass/fail and confidence | 80% |

The rule-based check is not a judge anyone could deploy without an answer key.
It is here as the ceiling for "a check with a known expected state", and its
misses say something about the truth itself.

### The conversations and the truth

230 conversations from the tau-bench retail test set: 115 tasks run by a
careful agent that confirms every change with the customer, and the same 115 by
a rushed agent instructed to act on the first version of a request. The agent
model is `openai/gpt-4.1` in both cases; the simulated customer is
`openai/gpt-4o-mini`.

Ground truth is tau-bench's deterministic check: the final database state must
match the expected state and any required output strings must appear in the
agent's messages. It says nothing about how the agent got there.

| | Pass | Fail | Pass rate |
| --- | ---: | ---: | ---: |
| Careful agent | 59 | 56 | 51.3% |
| Rushed agent | 54 | 61 | 47.0% |
| Both | 113 | 117 | 49.1% |

The classes are balanced, so accuracy of 0.49 (always pass) or 0.51 (always
fail) is the floor.

### The reading copy and the questions

Each judge reads the same text: the customer's task instruction, a summary of
the retail policy, and every turn of the conversation including tool calls and
tool results, whole. The median copy is about 3,000 tokens; the largest is
about 6,200. Nothing is cut unless a copy exceeds 28,000 tokens, and none does.

Each judge answers two questions in one call: how strongly "The agent completed
the user request in line with the policy" holds, on a 0 to 1 scale, and a
pass/fail choice against a rubric whose pass bar is: authenticated the user,
performed the requested changes correctly, confirmed before any order-changing
write, and gave correct final information. Fail conditions include writes
without confirmation and "stating information that is fabricated or unsupported
by the tools".

Every judge answered five times per conversation. The tables below score the
modal verdict of the five unless stated otherwise.

## Headline comparison

Both agents, full reading copy, modal verdict of five repeats, n = 230.

| Judge | Accuracy (95% CI) | Kappa vs truth | AUROC ("completed" score) | Fail recall | Fail precision | Judge pass rate | Unanimous across 5 | USD per verdict | Latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based check | 0.787 (0.73–0.84) | 0.57 | 0.79 | 0.92 | 0.73 | 37% | 100% | 0 | instant |
| Jev | 0.565 (0.50–0.63) | 0.14 | 0.73 (0.66–0.79) | 0.35 | 0.63 | 72% | 99% | $0.00019 | 0.17 s / 0.28 s |
| Fast text model | 0.565 (0.50–0.63) | 0.14 | 0.60 (0.54–0.67) | 0.19 | 0.82 | 88% | 96% | $0.00065 | 2.1 s / 3.2 s |
| Strong text model | 0.604 (0.54–0.67) | 0.21 | 0.69 (0.63–0.76) | 0.47 | 0.66 | 64% | 80% | $0.0280 | 28 s / 60 s |

Truth pass rate is 49%. AUROC is the chance that the judge's mean "completed"
score across repeats ranks a randomly chosen passing conversation above a
randomly chosen failing one; 0.5 is guessing. On the pass probability instead
the figures are Jev 0.71, gpt-4o-mini 0.59, gpt-5 0.68.

Paired bootstrap on the same 230 conversations:

| Comparison | Accuracy difference (95% CI) | AUROC difference (95% CI) |
| --- | ---: | ---: |
| gpt-5 vs Jev | +0.04 (−0.02 to +0.10) | −0.03 (−0.09 to +0.03) |
| gpt-5 vs gpt-4o-mini | +0.04 (−0.04 to +0.12) | +0.09 (+0.01 to +0.17) |
| Jev vs gpt-4o-mini | 0.00 (−0.07 to +0.07) | +0.12 (+0.04 to +0.21) |
| Rule-based check vs gpt-5 | +0.18 (+0.10 to +0.27) | |

Jev and gpt-5 cannot be told apart on either measure. Both beat gpt-4o-mini on
ranking. The rule-based check beats all three by a wide margin.

## Finding 1: the model judges are a little better than guessing

Split by agent:

| Judge | Careful agent accuracy | Rushed agent accuracy |
| --- | ---: | ---: |
| Rule-based check | 0.757 | 0.817 |
| Jev | 0.557 | 0.574 |
| Fast text model | 0.557 | 0.574 |
| Strong text model | 0.600 | 0.609 |

On the careful agent, where every write was confirmed and the process was
clean, the truth is decided entirely by whether the final state was right.
gpt-5 reaches 60% there; the other two are within six points of always-pass.
This is the cleanest test of whether a reading judge can tell a correct outcome
from an incorrect one, and the answer is: somewhat, for the strongest model.

Confusion matrices (rows are truth, columns are the judge's modal verdict):

| Rule-based check | fail | pass |
| --- | ---: | ---: |
| truth fail | 107 | 10 |
| truth pass | 39 | 74 |

| Jev | fail | pass |
| --- | ---: | ---: |
| truth fail | 41 | 76 |
| truth pass | 24 | 89 |

| Fast text model | fail | pass |
| --- | ---: | ---: |
| truth fail | 22 | 95 |
| truth pass | 5 | 108 |

| Strong text model | fail | pass |
| --- | ---: | ---: |
| truth fail | 55 | 62 |
| truth pass | 29 | 84 |

The rule-based check's 39 false fails come almost entirely from expected
read-only actions the agent never called (`get_order_details`, `calculate`,
`get_product_details`, `list_all_product_types`, `transfer_to_human_agents`).
A check restricted to state-changing actions would score higher. Its 10 false
passes are runs where every expected action was made and the state still did
not match, usually because of an extra write.

## Finding 2: equal accuracy, three different judges

Jev and gpt-4o-mini both score 0.565. They reach it in opposite ways.

gpt-4o-mini said pass on 203 of 230 conversations. Its fail recall is 0.19: it
let 95 of 117 failures through. Its edge over always-pass comes from the 22
failures it did catch, and it fails runs almost only when the agent visibly did
nothing. A gate that passes 88% of everything is not a gate.

Jev said pass on 165. It catches 41 of 117 failures. More usefully, its scores
sort conversations: AUROC 0.73, the best of the three. Its default threshold is
the problem, not its ranking. Choosing the threshold on held-out conversations
(5-fold cross-validation, 20 shuffles) raises its agreement with the truth to
68%, against 56% for gpt-4o-mini treated the same way and 66% for gpt-5.

gpt-5 said pass on 146, nearest the true 113. It catches 55 of 117 failures,
the best recall of the three, at the cost of 29 false fails. It is also the
least repeatable: 21 of 230 conversations split three to two across five
repeats, and single-repeat accuracy ranged from 0.591 to 0.635. The mode of
five (0.604) sits at the low end of that range, so repeating did not raise
accuracy; it made the verdict reproducible at five times the cost. For Jev and
gpt-4o-mini one call is as good as five.

## Finding 3: Jev and the strong text model judge process, not outcome

Kappa between judges' modal verdicts:

| | Rule-based | Jev | Fast text | Strong text | Truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule-based check | 1.00 | 0.04 | 0.04 | 0.06 | 0.57 |
| Jev | | 1.00 | 0.06 | 0.50 | 0.14 |
| Fast text model | | | 1.00 | 0.09 | 0.14 |
| Strong text model | | | | 1.00 | 0.21 |

Jev and gpt-5 agree with each other at kappa 0.50. Each agrees with the truth
at 0.14 and 0.21. Two judges built differently, a typed decision model with no
rationale and a reasoning text model, converge on the same verdicts, and those
are not the outcome verdicts.

What they converge on is process. The rushed agent's only change is that it no
longer confirms before acting. Its outcome pass rate falls 4 points. The judges'
pass rates fall much further, and they fall regardless of the outcome:

| Judge | Pass rate on truth-pass runs, careful → rushed | Pass rate on truth-fail runs, careful → rushed |
| --- | ---: | ---: |
| Truth | 1.00 → 1.00 | 0.00 → 0.00 |
| Rule-based check | 0.63 → 0.69 | 0.11 → 0.07 |
| Jev | 0.95 → 0.61 | 0.86 → 0.46 |
| Fast text model | 0.97 → 0.94 | 0.88 → 0.75 |
| Strong text model | 0.81 → 0.67 | 0.62 → 0.44 |

Jev's pass rate drops 34 points on runs that passed and 40 points on runs that
failed. It is reacting to the missing confirmation, which the rubric names as a
fail condition, not to whether the change was right. gpt-5's fail rationales
on rushed-agent runs that passed cite a missing confirmation 45% of the time,
against 16% on the careful agent.

This is why the regression experiment (gate 8) shows what it shows. The truth
gap between the two agents is −4.3 points. The rule-based check estimates −1.7
(interval −13.9 to +10.4), gpt-4o-mini −7.8 (−15.7 to 0.0), gpt-5 −17.4
(−29.6 to −5.2), and Jev −37.4 (−47.8 to −27.0). Jev's and gpt-5's intervals
exclude the truth. None of the four raised a false alarm when shown two halves
of the careful agent's own conversations.

Whether this is a defect depends on what you are measuring. If the question is
"did the agent reach the right state", these judges are wrong by a factor of
four to nine. If the question is "did the agent follow the policy", they are
the only judges here that noticed the policy was broken, and they noticed it
consistently. The rubric asked for both, and the ground truth checks only one;
the disagreement is between the rubric and the truth, and the judges sided with
the rubric.

## Finding 4: the failures every reading judge misses are detail errors

Forty-four conversations that failed the state check were passed by all three
model judges; 29 come from the careful agent. Comparing the agent's
state-changing tool calls against the task's expected ones:

| Why the state check failed | Count |
| --- | ---: |
| Right kind of write, wrong arguments (item, variant, payment method) | 21 |
| One of several requested writes never made | 8 |
| A required figure missing from the agent's messages | 5 |
| An extra write on top of the expected ones | 5 |
| Other | 5 |

An example. In `retail-106` the customer asks to exchange two identical tea
kettles for two different variants. The agent authenticates, reads the order,
lists the options, states the change and the price difference, gets an explicit
yes, and calls the exchange tool, with the same new variant twice. gpt-5's
rationale walks through every one of those steps approvingly and passes the
run. In `retail-101` the customer wants two items in a pending order modified;
the agent modifies one, on a payment method other than the expected one, and
is passed by all three judges with rationales that praise the confirmation
flow.

These are not failures of reasoning about the conversation. They are failures
to reconcile an identifier the customer implied against an identifier the agent
sent, a lookup that a reading judge does poorly and a line of code does
trivially. The rule-based check caught 40 of the 44.

Majority vote across the three model judges scores 0.565, kappa 0.14, no
better than the best single judge. Agreement between reading judges is not
evidence of correctness here.

## Finding 5: cutting the tool output was our mistake, and it changed the results

The first reading copy cut every tool result at 600 characters and marked the
cut. Across the 230 copies, 945 of 1,600 tool results (59%) were cut, removing
about 2,900 characters per conversation on average. The `truncated` flag on
each copy recorded only cuts made to fit the token budget, so it read false on
all 230 while most of their tool results were missing. The design document gave
no reason for the cap, and it saved little: whole, the copies average 3,000
tokens against 2,300 cut, nowhere near the 28,000 budget.

The judges were then asked whether the agent's statements were "supported by
the tools" against tool output that was not on the page. On careful-agent
conversations that passed the state check, gpt-5 gave 115 fail verdicts across
five repeats; 92% of the rationales cited unsupported or fabricated facts. Of
the 14 specific values it named, all 14 were in the raw tool results and 10
had been cut from what it read.

A 15-conversation gpt-5 sample on the uncut copies confirmed the cause before
the full rerun: of 12 conversations gpt-5 had failed 5/5 for unsupported facts,
6 flipped to pass on every repeat and one on three of five; the five that
stayed fail are now faulted for skipped confirmations or for policy rules the
agent stated that are not in the policy, which are legitimate rubric findings.
One of 3 control conversations that truly failed and gpt-5 had passed is now
caught.

All three model judges were re-run on the uncut copies:

| | Jev | gpt-4o-mini | gpt-5 |
| --- | ---: | ---: | ---: |
| AUROC ("completed"), cut → uncut | 0.65 → 0.73 | 0.58 → 0.60 | 0.58 → 0.69 |
| Paired AUROC gain (95% CI) | +0.04 to +0.11 | −0.03 to +0.08 | +0.06 to +0.18 |
| Accuracy, cut → uncut | 54% → 57% | 56% → 57% | 54% → 60% |
| Kappa, cut → uncut | 0.08 → 0.14 | 0.12 → 0.14 | 0.08 → 0.21 |
| Unanimous across 5, cut → uncut | 92% → 99% | 94% → 96% | 66% → 80% |
| Judge pass rate, cut → uncut | 51% → 72% | 87% → 88% | 49% → 64% |
| Fail recall, cut → uncut | 53% → 35% | 20% → 19% | 55% → 47% |

Three different responses to the same change. gpt-5 gained the most, on
ranking, accuracy, and repeatability, and its remaining "unsupported" findings
on correct runs are now about agent statements (a refund route the tool had
rejected, a return rule not in the policy), not about missing numbers. Jev
gained on ranking and became almost perfectly repeatable, but also became much
more willing to pass, so its default threshold slipped. gpt-4o-mini did not
move; the extra 950 input tokens per call bought nothing, which suggests it
reaches its verdict without reading the tool results closely.

Before the fix the headline read "gpt-5 is at chance and the least repeatable
judge". After it, gpt-5 is the most accurate model judge. The reading copy is
part of the judge.

## Finding 6: stated confidence is not usable as-is

Calibration on the uncut copies (gate 6), n = 230 per row. ECE is the average
gap between stated probability and observed frequency; Brier is mean squared
error of the probability, where a constant 0.5 scores 0.25.

| Judge | Signal | ECE | Brier | ECE after cross-fitted isotonic | Brier after |
| --- | --- | ---: | ---: | ---: | ---: |
| Rule-based check | pass probability | 0.21 | 0.21 | 0.05 | 0.17 |
| Jev | "completed" (0–1) | 0.21 | 0.26 | 0.13 | 0.22 |
| Jev | pass probability | 0.28 | 0.30 | 0.10 | 0.23 |
| Fast text model | "completed" (0–1) | 0.44 | 0.44 | 0.07 | 0.24 |
| Fast text model | pass probability | 0.36 | 0.38 | 0.05 | 0.26 |
| Strong text model | "completed" (0–1) | 0.22 | 0.28 | 0.12 | 0.23 |
| Strong text model | pass probability | 0.20 | 0.27 | 0.10 | 0.23 |

Every model judge's raw Brier is at or above the coin's 0.25; gpt-4o-mini's
0.44 is far worse than a coin because it states about 0.9 confidence whether
it is right or wrong. After isotonic correction the Brier scores land on 0.22
to 0.26, still near a coin. Recalibration fixes the gap but not the ranking.

The cascade experiment (gate 5) follows. Asking gpt-4o-mini first and
escalating to gpt-5 when its confidence is below 0.5 sends 26% of
conversations on, reaches 62% accuracy at $0.008 per conversation, and is the
best point on the cost curve: 71% cheaper than gpt-5 alone for one point more.
But it is also only five points over gpt-4o-mini alone at 12× the price, and
one point over Jev alone at 40× the price.

## Finding 7: cost and latency

Per verdict on the uncut copies, at list prices dated 2026-09-24, mean over
1,150 calls each:

| Judge | Input tokens | Output tokens | USD per verdict | USD per 1,000 | USD for 230 × 5 repeats | Latency p50 | Latency p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based check | 0 | 0 | 0 | 0 | 0 | 0 ms | 0 ms |
| Jev | 4,611 | 0 | $0.00019 | $0.19 | $0.22 | 169 ms | 280 ms |
| Fast text model | 3,941 | 100 | $0.00065 | $0.65 | $0.75 | 2,086 ms | 3,204 ms |
| Strong text model | 3,936 | 2,309 | $0.0280 | $28.01 | $32.21 | 28,336 ms | 59,729 ms |

The full-copy judging cost $33.18 across the three model judges; gpt-5
accounts for $32.21 of it. Its output is almost all reasoning tokens, which is
also why a verdict takes half a minute. Jev's latency includes the OpenRouter
gateway.

For the same ranking quality, Jev is about 150 times cheaper and 170 times
faster than gpt-5, and it gives a reproducible verdict from one call where
gpt-5 needs three or five.

## What this means for choosing a judge

| If you need to | Use | Because | But |
| --- | --- | --- | --- |
| Know whether the agent reached the correct end state | A check in code against an expected state or a structured diff | It is the only thing here that caught outcome errors (0.79, fail recall 0.92, free, instant) | It needs an answer key per task, and a naive one penalizes valid alternative paths; check state-changing actions, not the whole expected path |
| Audit whether the agent followed the process (authenticated, confirmed, stayed in policy) | Jev | It ranks conversations as well as gpt-5 (kappa 0.50 between them) at $0.19 per thousand and 0.17 s, with the same verdict from one call 99% of the time | Its default threshold leans lenient on whole copies; set it on conversations you have labelled. It will report a process regression as a large outcome regression |
| The same audit, with a written explanation per verdict and the best default pass/fail line | gpt-5, three to five repeats | Most accurate model judge (60%), best fail recall (47%), and a rationale a person can act on | 80% single-call reproducibility, 28 s and 3 cents per call, and it is the judge most sensitive to what the reading copy leaves out |
| Catch failures cheaply as a gate | Not gpt-4o-mini | Fail recall 0.19; it passes 88% of everything, and giving it more evidence changed nothing | Usable as a very weak first filter and nothing more |
| Track a regression between two agent versions | Whichever judge you use, run it on both versions and compare with an interval | None of the four false-alarmed on a split of the same agent | Process-sensitive judges exaggerate: −37 and −17 points reported for a −4 truth. The direction was right for every judge that detected anything |
| Route items by confidence, or report a probability | Recalibrate on held-out conversations first, then expect little | Raw Brier is at or above a coin for all three model judges | Even recalibrated, none separates right from wrong verdicts well; a cascade gated on confidence gained one or two points over a single judge |
| Any of the above | Give the judge the whole tool output | 59% of tool results were cut in our first copy; that created false fails in the strictest judge, hid the identifiers every judge needs to catch real failures, and reordered the judges | Whole copies cost about 30% more input tokens; at Jev's and gpt-4o-mini's prices that is negligible, at gpt-5's it is about a cent a call |

Two general points fall out of this.

First, decide which truth you want before you pick a judge. Outcome truth
(final state) and process truth (policy followed) are different measurements,
and this experiment shows reading judges measure the second even when asked
for the first. If you grade against outcome truth, a process-sensitive judge
will look worse than it is; if you grade against a policy rubric, gpt-4o-mini
will look better than it is.

Second, the reading copy is part of the judge. Finding 5 is the largest single
effect in this study, and it was caused by a serialization choice, not by any
model. Before comparing judges, check what your serializer removed, and whether
the rubric asks the judge to verify things that are no longer on the page.

## Limitations

- One domain (tau-bench retail), one agent model, two policy variants, 115
  tasks. Confidence intervals on accuracy are about ±6.5 points at n = 230;
  differences among Jev and gpt-5 are inside that.
- The ground truth is tau-bench's strict state match, a proxy for task success.
  It penalizes valid alternative paths and does not measure process. Part of
  what reads as judge error here is a rubric that asks for process against a
  truth that ignores it.
- The simulated customer is itself a small model and sometimes departs from its
  script; some ground-truth failures are the customer's, not the agent's, and
  no judge can be expected to catch those.
- The rule-based check reads the answer key. It is a ceiling, not a deployable
  judge.
- Text judges ran at provider defaults with one shared rubric and no per-judge
  prompt tuning. A rubric that separated outcome from process would likely
  change the process-sensitive judges' regression estimates.
- Jev's training data is described by its vendor as synthetic and cannot be
  inspected. Jev was served through OpenRouter, so its latency includes the
  gateway. Kayzn has no relationship with TypeSafe, OpenAI, or OpenRouter.
- Prices are list prices on the date in `config/pricing.toml`.
- One random seed per stage; variance across seeds is not measured.
- The cut-copy verdicts remain in `cache/judge` under their old state hashes
  and are the basis of the before/after comparison in Finding 5. They are not
  used anywhere else.

## Reproducing the numbers

The gate tables are committed under `results/` and can be rebuilt from the
verdict cache without any paid call:

```
judges analyze --gate g3 --profile full --profile compact   # results/g3_summary.md
judges analyze --gate g5                                    # results/g5_*.md
judges analyze --gate g6                                    # results/g6_*.md
judges analyze --gate g8                                    # results/g8_*.md
```

The rows for the four judges on the `full` profile in `g3_summary.md` are the
headline accuracy, kappa, F1, agreement, token, and latency figures. Gate 6
gives the calibration table, gate 8 the regression estimates, gate 5 the
cascade.

The remaining figures (confusion matrices, judge-to-judge kappa, per-agent pass
rates split by truth, single-repeat versus modal accuracy, the miss taxonomy,
rationale themes, and the cost table) are printed by one script that reads the
same caches:

```
uv run python scripts/judge_comparison.py
```

It takes the modal verdict of the five repeats per judge and conversation,
joins on `state_hash` to the full-profile states, and compares the agent's
state-changing tool calls with the task's expected actions using the same
normalization as the rule-based check. The before/after figures in Finding 5
pair each conversation's verdicts on the cut copy (state hashes from commit
`fc7e935`) with its verdicts on the uncut copy. The bootstrap intervals, the
cross-validated thresholds, and the check of the "unsupported" values against
raw tool output were one-off computations on the same data and are described
in the text where they appear.
