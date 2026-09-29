# Choosing a judge: what 4,600 verdicts say

An analysis of the four judges that read the full conversation text, written
for someone deciding what kind of judge to put in an agent evaluation loop.
The local model Laya and the short reading copy made for it are left out: Laya
judged only 69 conversations before its run was interrupted, and the short
copy exists only to fit Laya's 512-token limit, so neither adds to a comparison
of judges that can read a whole conversation.

Analysis date: 2026-09-28. Everything below is computed from the committed
caches; see [Reproducing the numbers](#reproducing-the-numbers).

## Summary

1. Against tau-bench's outcome ground truth, the three model judges are at
   chance. Jev, the fast text model, and the strong text model score 0.54, 0.56,
   and 0.54 on a task where always answering pass scores 0.49. None of the three
   is distinguishable from the others, and only the fast text model is
   distinguishable from always-pass, by two to eleven points. The rule-based
   check, which reads the task's answer key, scores 0.79.

2. Equal accuracy hides three different judges. The fast text model passes 87%
   of everything and catches one failure in five. The strong text model is
   balanced but unstable: a third of conversations get split verdicts across
   five repeats. Jev is balanced, gives the same verdict 92% of the time, and is
   the fastest and cheapest of the three by a wide margin.

3. Jev and the strong text model agree with each other far more than with the
   truth (kappa 0.53 between them, 0.08 with the truth). They are judging the
   same thing, and it is not final state: it is whether the agent followed the
   process the rubric describes. When the rushed agent stops confirming before
   it acts, both judges fail it about 30 points more often, whether or not the
   final state was right. They report a 20 to 34 point regression where the
   truth shows 4.

4. The failures every reading judge misses are detail errors. Thirty-seven
   failed conversations were passed unanimously by all three model judges. In
   18 the agent made the right kind of change with the wrong item, variant, or
   payment method; in 8 it dropped one of several requested changes; in 5 it
   omitted a required figure. These conversations look procedurally clean.
   Catching them means cross-checking identifiers against tool output, which
   no judge here did.

5. The reading copy caused false fails. Fifty-nine percent of tool results in
   the full copy were cut at 600 characters. Of the strong text model's fail
   verdicts on careful-agent conversations that actually passed, 92% cite
   information "unsupported by tool output". Every one of the 14 specific values
   it named exists in the raw tool results; 10 had been cut from the copy the
   judge read. A strict judge applied the rubric's fabrication rule correctly to
   evidence the serializer had removed.

6. No judge's stated confidence is usable as a probability. Every model judge's
   Brier score on pass probability is at or above 0.25, the score of a coin. A
   cross-fitted recalibration removes most of the miscalibration but does not
   add discrimination. A cascade gated on the fast model's confidence bought
   nothing over the fast model alone.

7. Cost and latency differ by two orders of magnitude for the same accuracy. Per
   thousand verdicts: Jev $0.14 at 0.25 s median, fast text model $0.51 at
   1.9 s, strong text model $32.64 at 36 s. The strong model spent 98% of the
   study's judging budget.

8. For someone choosing: if you can write down the expected end state, check it
   in code. If you cannot, a reading judge will audit process, not outcome; Jev
   does that audit as well as the strong text model at a small fraction of the
   cost, and the fast text model does not do it at all. Whatever you choose, do
   not truncate the tool output the judge is asked to verify against, and do
   not act on the probabilities without recalibrating them on held-out data.

## What was measured

### The judges

| Judge | What it is | What it sees | How it answers | Deterministic |
| --- | --- | --- | --- | --- |
| Rule-based check | Code that compares the agent's tool calls and final message to the task's expected actions and required outputs | The raw trajectory and the task's answer key | pass/fail at confidence 1.0 | Yes |
| Jev | `jev-1.13`, a hosted typed decision model, served through OpenRouter | The reading copy | Probabilities for the two typed questions; no free text | Nearly: 92% of conversations get five identical verdicts |
| Fast text model | `openai/gpt-4o-mini` | The reading copy and the rubric | A written rationale plus a structured pass/fail and confidence | 94% identical verdicts, mostly because it almost always says pass |
| Strong text model | `openai/gpt-5` | The reading copy and the rubric | A written rationale (about 2,900 output tokens) plus structured pass/fail and confidence | 66% identical verdicts |

The rule-based check is not a judge anyone could deploy without an answer key.
It is here as the ceiling for "a check with a known expected state", and its
misses are informative about the truth itself.

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
tool results. At the time of this analysis the serializer cut each tool result
at 600 characters, and the median copy was about 2,200 tokens. That cap has
since been removed (see Finding 5); the current full copy carries every tool
result whole and runs about 3,000 tokens. The verdicts analyzed here were all
judged on the cut copies.

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

| Judge | Accuracy (95% CI) | Kappa vs truth | AUROC (pass prob.) | Fail recall | Fail precision | Judge pass rate | Unanimous across 5 | USD per verdict | Latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based check | 0.787 (0.73–0.84) | 0.57 | 0.79 | 0.92 | 0.73 | 37% | 100% | 0 | instant |
| Jev | 0.539 (0.48–0.60) | 0.08 | 0.59 | 0.53 | 0.55 | 51% | 92% | $0.00014 | 0.25 s / 0.44 s |
| Fast text model | 0.557 (0.49–0.62) | 0.12 | 0.62 | 0.20 | 0.74 | 87% | 94% | $0.00051 | 1.9 s / 3.1 s |
| Strong text model | 0.539 (0.48–0.60) | 0.08 | 0.56 | 0.55 | 0.55 | 49% | 66% | $0.0326 | 36 s / 75 s |

Truth pass rate is 49%. AUROC uses the mean pass probability across repeats;
Jev's answer to the "completed" question does better (0.65) than its pass/fail
probability (0.59), which is the highest discrimination any model judge shows.

Paired bootstrap on the same 230 conversations: the accuracy difference between
any two of the three model judges has a 95% interval that includes zero (the
widest is −0.07 to +0.10). Against an always-pass judge, only the fast text
model's margin excludes zero (+0.02 to +0.11); Jev's and the strong model's run
from −0.04 to +0.14.

## Finding 1: the model judges are at chance against outcome truth

Split by agent:

| Judge | Careful agent accuracy | Rushed agent accuracy |
| --- | ---: | ---: |
| Rule-based check | 0.757 | 0.817 |
| Jev | 0.539 | 0.539 |
| Fast text model | 0.548 | 0.565 |
| Strong text model | 0.504 | 0.574 |

On the careful agent, where every write was confirmed and the process was
clean, the strong text model is at exactly chance and the other two are within
five points of it. This is the cleanest test of whether a reading judge can
tell a correct outcome from an incorrect one, and none of them can.

Confusion matrices (rows are truth, columns are the judge's modal verdict):

| Rule-based check | fail | pass |
| --- | ---: | ---: |
| truth fail | 107 | 10 |
| truth pass | 39 | 74 |

| Jev | fail | pass |
| --- | ---: | ---: |
| truth fail | 62 | 55 |
| truth pass | 51 | 62 |

| Fast text model | fail | pass |
| --- | ---: | ---: |
| truth fail | 23 | 94 |
| truth pass | 8 | 105 |

| Strong text model | fail | pass |
| --- | ---: | ---: |
| truth fail | 64 | 53 |
| truth pass | 53 | 60 |

The rule-based check's 39 false fails come almost entirely from expected
actions the agent never called: `get_order_details` (9), `calculate` (8),
`get_product_details` (5), `list_all_product_types` (4),
`transfer_to_human_agents` (4). These are read-only steps on the answer key's
path that the agent skipped while still reaching the right state. A check
restricted to state-changing actions would score higher. Its 10 false passes
are runs where every expected action was made and the state still did not
match, usually because of an extra write.

## Finding 2: equal accuracy, three different judges

The fast text model's 0.557 is the best accuracy of the three and the least
meaningful. It said pass on 199 of 230 conversations. Its fail recall is 0.20:
it let 94 of 117 failures through. Its edge over always-pass comes from the 23
failures it did catch, and its accuracy falls to 0.28 on the longest third of
conversations, where the truth pass rate is 0.24 and it keeps saying pass. A
gate that passes 87% of everything is not a gate.

The strong text model's verdicts are balanced (49% pass) and unstable. Only 152
of 230 conversations got the same verdict five times; 38 split three to two.
Its single-repeat accuracy ranged from 0.509 to 0.565 across the five runs.
Taking the mode of five did not raise accuracy (0.539 modal against a 0.541
single-repeat mean); it only made the verdict reproducible, at five times the
cost.

Jev's verdicts are balanced (51% pass) and stable: 211 of 230 conversations
unanimous, 8 split three to two, single-repeat accuracy between 0.535 and
0.557. For Jev and the fast text model, one call is as good as five.

## Finding 3: Jev and the strong text model judge process, not outcome

Kappa between judges' modal verdicts:

| | Rule-based | Jev | Fast text | Strong text | Truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule-based check | 1.00 | 0.04 | 0.06 | −0.02 | 0.57 |
| Jev | | 1.00 | 0.08 | 0.53 | 0.08 |
| Fast text model | | | 1.00 | 0.06 | 0.12 |
| Strong text model | | | | 1.00 | 0.08 |

Jev and the strong text model agree on 76.5% of conversations, kappa 0.53. Each
agrees with the truth at kappa 0.08. Two judges built differently, one a typed
decision model with no rationale, one a reasoning text model, converge on the
same verdicts, and those verdicts are not the outcome verdicts.

What they converge on is the process. The rushed agent's only change is that
it no longer confirms before acting. Its outcome pass rate falls 4 points. The
judges' pass rates fall much further, and they fall regardless of the outcome:

| Judge | Pass rate on truth-pass runs, careful → rushed | Pass rate on truth-fail runs, careful → rushed |
| --- | ---: | ---: |
| Truth | 1.00 → 1.00 | 0.00 → 0.00 |
| Rule-based check | 0.63 → 0.69 | 0.11 → 0.07 |
| Jev | 0.71 → 0.37 | 0.64 → 0.31 |
| Fast text model | 0.95 → 0.91 | 0.88 → 0.74 |
| Strong text model | 0.59 → 0.46 | 0.59 → 0.33 |

Jev's pass rate drops 34 points on runs that passed and 33 points on runs that
failed. It is reacting to the missing confirmation, which the rubric names as a
fail condition, not to whether the change was right. The strong text model's
fail rationales on rushed-agent runs that passed cite a missing confirmation
42% of the time.

This is why the regression experiment (gate 8) shows what it shows. The truth
gap between the two agents is −4.3 points. The rule-based check estimates −1.7
(interval −13.9 to +10.4), the fast text model −9.6 (−18.3 to −0.9), the strong
text model −20.0 (−33.0 to −7.0), and Jev −33.9 (−46.1 to −21.7). Jev's and the
strong model's intervals exclude the truth. None of the four raised a false
alarm when shown two halves of the careful agent's own conversations.

Whether this is a defect depends on what you are measuring. If the question is
"did the agent reach the right state", these judges are wrong by a factor of
five to eight. If the question is "did the agent follow the policy", they are
the only judges here that noticed the policy was broken, and they noticed it
consistently. The rubric asked for both, and the ground truth checks only one;
the disagreement is between the rubric and the truth, and the judges sided with
the rubric.

## Finding 4: the failures every reading judge misses are detail errors

Forty conversations were misjudged by all three model judges. Thirty-seven of
them are failures that every model judge passed; 27 come from the careful
agent. Comparing the agent's state-changing tool calls against the task's
expected ones:

| Why the state check failed | Count |
| --- | ---: |
| Right kind of write, wrong arguments (item, variant, payment method) | 18 |
| One of several requested writes never made | 8 |
| A required figure missing from the agent's messages | 5 |
| An extra write on top of the expected ones | 3 |
| Other | 3 |

An example. In `retail-106` the customer asks to exchange two identical tea
kettles for two different variants. The agent authenticates, reads the order,
lists the options, states the change and the price difference, gets an explicit
yes, and calls the exchange tool, with the same new variant twice. The strong
text model's rationale walks through every one of those steps approvingly and
passes the run. In `retail-101` the customer wants two items in a pending order
modified; the agent modifies one, on a payment method other than the expected
one, and is passed by all three judges with rationales that praise the
confirmation flow.

These are not failures of reasoning about the conversation. They are failures
to reconcile an identifier the customer implied against an identifier the agent
sent, a lookup that a reading judge does poorly and a line of code does
trivially. The rule-based check caught 34 of the 37.

Majority vote across the three model judges scores 0.565, kappa 0.13, and the
96 conversations where all three agreed were judged right 58% of the time.
Agreement between reading judges is not evidence of correctness here.

## Finding 5: the reading copy caused false fails

When these verdicts were judged, the full reading copy cut each tool result at
600 characters and marked the cut. Across the 230 copies, 945 of 1,600 tool
results (59%) were cut, removing about 2,900 characters per conversation on
average. Only 7 copies were untouched. The cap has since been removed and the
copies re-serialized whole; Jev has been re-judged on the uncut copies (see the
end of this section), the other judges have not yet.

On careful-agent conversations that passed the state check, the strong text
model gave 115 fail verdicts across the five repeats, touching 34 of the 59
conversations (24 of them by modal verdict). Their rationales cite:

| Theme in the fail rationale | Share |
| --- | ---: |
| Fabricated or unsupported statement | 92% |
| Policy or eligibility violation | 63% |
| Authentication | 23% |
| Missing confirmation | 20% |
| Wrong or missing action | 17% |

Typical sentences: "provided a specific tracking number (907614527588) for
order #W5565470 without any visible tool output supporting that number";
"stated a specific refund amount ($60.78) after the exchange without any tool
output supporting that calculation". Checking the 14 specific values the judge
named against the data: all 14 appear in the raw tool results the agent
actually received, and 10 of the 14 had been cut from the tool lines of the
copy the judge read. The agent did not fabricate. The judge, applying the
rubric's fabrication rule to the evidence in front of it, was right about what
it could see and wrong about the run.

The fast text model rarely fails a run for this reason (11% of its few false
fails), because it rarely fails a run. Jev gives no rationale, so its share of
this effect cannot be read directly, but its false-fail rate on careful-agent
passes (29%) sits between the other two.

The same truncation blinds every judge to the detail errors in Finding 4: the
item and variant identifiers needed to catch them live in the tool results that
were cut.

Re-judging Jev on the uncut copies (230 conversations, 5 repeats, $0.22)
confirms the cap was costing it. Its AUROC on the "completed" score rose from
0.65 to 0.73 (95% CI 0.66 to 0.79; the paired gain is +0.04 to +0.11),
agreement with the truth at its own cut-off rose from 54% to 57%, and the share
of conversations with five identical verdicts rose from 92% to 99%. Its pass
rate also rose, from 51% to 72%, against a true rate of 49%, so a cut-off
chosen on held-out conversations (about 68% agreement under 5-fold cross
validation) does better than its default. It still grades the rushed agent's
skipped confirmations: pass rate 90% on the careful agent, 53% on the rushed
one, where the truth moves from 51% to 47%.

## Finding 6: stated confidence is not usable as-is

Calibration on the full copy (gate 6), n = 230 per row. ECE is the average gap
between stated probability and observed frequency; Brier is mean squared error
of the probability, where a constant 0.5 scores 0.25.

| Judge | Signal | ECE | Brier | ECE after cross-fitted isotonic | Brier after |
| --- | --- | ---: | ---: | ---: | ---: |
| Rule-based check | pass probability | 0.21 | 0.21 | 0.04 | 0.17 |
| Jev | "completed" (0–1) | 0.16 | 0.26 | 0.09 | 0.24 |
| Jev | pass probability | 0.24 | 0.30 | 0.08 | 0.26 |
| Fast text model | "completed" (0–1) | 0.44 | 0.43 | 0.04 | 0.25 |
| Fast text model | pass probability | 0.35 | 0.37 | 0.11 | 0.24 |
| Strong text model | "completed" (0–1) | 0.22 | 0.30 | 0.04 | 0.25 |
| Strong text model | pass probability | 0.21 | 0.30 | 0.04 | 0.25 |

Every model judge's raw Brier score is at or above the coin's 0.25; the fast
text model's 0.43 is far worse than a coin because it states 0.9 confidence on
runs it gets wrong as often as on runs it gets right (mean confidence 0.904
when right, 0.907 when wrong). The strong text model is the same (0.79 versus
0.78). Jev's confidence carries a little signal (0.59 when right, 0.50 when
wrong), which is where its slightly better "completed" AUROC comes from.

Recalibration fixes the gap but not the ranking: after isotonic correction the
Brier scores land on 0.24 to 0.26, still a coin. A calibrated coin is still a
coin.

The cascade experiment (gate 5) follows. Asking the fast model first and
escalating to the strong model when its confidence is below a threshold
produces 0.548 accuracy at $0.0155 per conversation at the best threshold,
against 0.557 at $0.0005 for the fast model alone and 0.539 at $0.0326 for the
strong model alone. Escalating on a confidence that does not know when it is
wrong cannot route the hard cases anywhere useful.

## Finding 7: cost and latency

Per verdict on the full copy, at list prices dated 2026-09-24, mean over 1,150
calls each:

| Judge | Input tokens | Output tokens | USD per verdict | USD per 1,000 | USD for 230 × 5 repeats | Latency p50 | Latency p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-based check | 0 | 0 | 0 | 0 | 0 | 0 ms | 0 ms |
| Jev | 3,367 | 0 | $0.00014 | $0.14 | $0.16 | 251 ms | 438 ms |
| Fast text model | 2,991 | 102 | $0.00051 | $0.51 | $0.59 | 1,900 ms | 3,057 ms |
| Strong text model | 2,986 | 2,891 | $0.0326 | $32.64 | $37.54 | 36,218 ms | 74,539 ms |

The judging step cost $58.94 in total across both reading copies; the strong
text model accounts for $57.92 of it. Its output is almost all reasoning
tokens, which is also why a verdict takes half a minute. Jev's latency includes
the OpenRouter gateway; a direct route would be lower.

For the same accuracy, Jev is 230 times cheaper and 140 times faster than the
strong text model, and it gives a reproducible verdict from one call where the
strong model needs three or five.

## What this means for choosing a judge

| If you need to | Use | Because | But |
| --- | --- | --- | --- |
| Know whether the agent reached the correct end state | A check in code against an expected state or a structured diff | It is the only thing here that caught outcome errors (0.79, fail recall 0.92, free, instant) | It needs an answer key per task, and a naive one penalizes valid alternative paths; check state-changing actions, not the whole expected path |
| Audit whether the agent followed the process (authenticated, confirmed, stayed in policy) | Jev | It tracks process violations as consistently as the strong text model (kappa 0.53 between them) at $0.14 per thousand and 0.25 s, with a stable verdict from one call | It will report a process regression as a large outcome regression; do not read its pass rate as task success |
| The same audit, with a written explanation per verdict | Strong text model, three to five repeats | It produces a rationale a person can act on and is balanced between pass and fail | 66% single-call reproducibility, 36 s and 3 cents per call, and it fails runs for "unsupported" facts whenever the copy hides the tool output |
| Catch failures cheaply as a gate | Not the fast text model | Fail recall 0.20; it passes 87% of everything | Its 0.557 "best accuracy" is a base-rate artifact; it is usable only as a very weak filter |
| Track a regression between two agent versions | Whichever judge you use, run it on both versions and compare with an interval | None of the four false-alarmed on a split of the same agent | Process-sensitive judges exaggerate: −34 and −20 points reported for a −4 truth. The direction was right for every judge that detected anything |
| Route items by confidence, or report a probability | Recalibrate on held-out conversations first, then expect little | Raw Brier is at or above a coin for all three model judges | Even recalibrated, none of them separates right from wrong verdicts; a cascade gated on confidence bought nothing |
| Any of the above | Give the judge the full tool output | 59% of tool results were truncated; that created false fails in strict judges and hid the identifiers every judge would need to catch real failures | Longer copies cost more per call; at Jev's and the fast model's prices that is negligible, at the strong model's it is not |

Two general points fall out of this.

First, decide which truth you want before you pick a judge. Outcome truth
(final state) and process truth (policy followed) are different measurements,
and this experiment shows reading judges measure the second even when asked for
the first. If you grade against outcome truth, a process-sensitive judge will
look worse than it is; if you grade against a policy rubric, the fast text
model will look better than it is.

Second, the reading copy is part of the judge. Two of the seven findings above
are about what the judge could see, not what it could do. Before comparing
models, check what the serializer removed, and whether the rubric asks the
judge to verify things that are no longer on the page.

## Limitations

- One domain (tau-bench retail), one agent model, two policy variants, 115
  tasks. Confidence intervals on accuracy are about ±6.5 points at n = 230;
  differences among the three model judges are inside that.
- The ground truth is tau-bench's strict state match, a proxy for task success.
  It penalizes valid alternative paths and does not measure process. Part of
  what reads as judge error here is a rubric that asks for process against a
  truth that ignores it.
- The simulated customer is itself a small model and sometimes departs from its
  script; some ground-truth failures are the customer's, not the agent's, and
  no judge can be expected to catch those.
- The rule-based check reads the answer key. It is a ceiling, not a deployable
  judge.
- Tool results in the reading copy were cut at 600 characters when these
  verdicts were judged. This is a finding about serialization and a confound
  for every judge comparison here. The cap has since been removed.
- Text judges ran at provider defaults with one shared rubric and no per-judge
  prompt tuning. A rubric that separated outcome from process, or told the
  judge that tool output was abridged, would likely change the strong text
  model's false-fail rate.
- Jev's training data is described by its vendor as synthetic and cannot be
  inspected. Jev was served through OpenRouter, so its latency includes the
  gateway.
- Prices are list prices on the date in `config/pricing.toml`.
- One random seed per stage; variance across seeds is not measured.

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
rationale themes, truncation counts, and the cost table) are printed by one
script that reads the same caches:

```
uv run python scripts/judge_comparison.py
```

It takes the modal verdict of the five repeats per judge and conversation,
joins on `state_hash` to the full-profile states, and compares the agent's
state-changing tool calls with the task's expected actions using the same
normalization as the rule-based check. Where a conversation has a duplicate
cached verdict for the same judge and repeat, the first file in path order is
kept. The bootstrap intervals in the headline section and the check of the 14
"unsupported" values against raw tool output were one-off computations on the
same data and are described in the text where they appear.
