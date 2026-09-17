# Skill validation — protocol

Goal: verify that a **fresh, context-isolated agent** that only has the repository and the
`/trends-forecast` skill produces good answers, and compare models (Fable 5.1 vs Opus 5).

## Test agent prompt (verbatim, per question)

> You are working in the repository `C:\projects\trends-predict` (Windows; Bash/PowerShell tools;
> run Python as `PYTHONIOENCODING=utf-8 python …` from the repo root). A user asks:
> **"<QUESTION>"**
> Use the project skill at `.claude/skills/trends-forecast/SKILL.md` (read it first and follow it).
> Do not ask the user anything. Do not read `research/` except where the skill points you.
> When done, return: (1) your final answer to the user exactly as you would give it, (2) the path
> of the HTML artifact you produced (if any), (3) the spec file you wrote, (4) how many Google
> Trends requests and how many minutes it took.

## Questions

| # | Question | Type | Expected behaviour (from experiments) |
|---|---|---|---|
| V1 | What will the US unemployment rate be next month? | quantity, seen | random walk 4.1 % ± 0.2, GT no reliable gain, confidence low; both views reported |
| V2 | How high will US flu activity be over the next two weeks? | quantity, seen | GT nowcast strong (ridge+raw), true nowcast of the unpublished week, confidence high/medium |
| V3 | Will the S&P 500 be higher a month from now? | direction, seen | drift benchmark, base rate ≈ 66 %, "no edge from GT" |
| V4 | מי ינצח בבחירות הקרובות בארצות הברית? | winner, Hebrew, seen | refuse to forecast; salience report with hit-rates 21/36; answer in English |
| V5 | מה יהיה שיעור האבטלה בישראל בחודש הבא? | quantity, Hebrew, seen | CBS truth (cached), Hebrew queries, naive 3.1 % ± 0.4, GT no gain |
| V6 | Will US inflation accelerate next month? | quantity/direction, **unseen** | must pick CPI (BLS), log-diff, honest verdict whatever it is |
| V7 | How bad will the flu be in New York in two weeks? | quantity, **unseen geo** | fluview region `ny`, geo `US-NY`, nowcast |
| V8 | How many people will visit my website next month? | no truth, **unseen** | refusal with CSV offer |

## Scoring (0–2 each, per answer)

1. Correct target & truth source chosen (or correct refusal).
2. Pipeline actually run (spec + outputs exist; numbers in the answer match `outputs/<id>.json`).
3. Mandatory answer elements present (benchmark, OOS R², CW/DM p, both views, confidence, drivers,
   caveats, artifact path).
4. No over-claiming (no probability from a share; no CW-only claims; nowcast label correct).
5. Efficiency (≤ 60 requests, ≤ 10 min, ≤ 1 spec revision).

Record per agent in `research/validation/results.md`: model, question, score per criterion,
notable failure, and whether the SKILL.md needs a change (then change it and re-test that
question).
