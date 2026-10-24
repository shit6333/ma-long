# E4 — Does inst-LC improve the instance MAP, not just the trajectory?

**Written 2026-09-03 from the OpenSe3r side.** Small experiment, one new arm.

## 0. What is already proven — do not re-run it

`docs/RESULTS.md` establishes **inst-LC → better ATE** and does it well: 5-run means,
false-loop counts audited against GT co-visibility, verdicts 15/15 on TUM and 20/20 on
Clio. **That is contribution #3 and it is finished.** This document does not ask you to
touch it.

## 1. The question

The paper's causal chain ends with `improved trajectory → better map`. That arrow is
currently **asserted, not measured**, and two things break the "it follows from ATE"
assumption:

1. **Different sequences.** The ATE evidence is on TUM / Clio (trajectory benchmarks).
   The map is evaluated on ScanNet / SceneNN / 3RScan / ScanNet++. We have never shown
   the LC gain on the sequences we actually build maps on.
2. **ATE is a trajectory average; a map cares about specific revisits.** Mean error can
   fall while the one revisit that mattered stays misaligned — or barely move while
   three revisits snap into place. They are not two views of the same quantity.

The mechanism being tested: an object seen at t=0 and again at t=500 lands in two
different places when the pose has drifted, the geometric cue then says "far apart", and
the map ends up with **two instances of one object**. Loop closure should collapse them.

## 2. Why the answer is interesting in **either** direction

This is the reason it is worth an afternoon:

* **If AP barely moves** → the identity descriptor re-identified the object *despite*
  the drift. That is the most direct evidence we have for the paper's central claim, and
  it comes almost free.
* **If AP moves a lot** → those duplicates were geometric, and the descriptor did not
  rescue them. Useful to know, and it makes the LC contribution stronger, but it weakens
  the "descriptor carries identity" story.

Either way we learn something. There is no outcome that wastes the run.

## 3. Design

**Only the "LC off" arm is new** — the regime-C runs in `outputs/ma_sem/<dataset>_final/`
already have inst-LC ON (`docs/RESULTS.md` protocol B). Re-run those same scenes with the
loop verifier disabled, everything else byte-identical (same checkpoint `a1_pq`, same
`conf_gate_coef 0.5`, same 25 px keyframer, same seed).

**Two datasets are enough**, chosen by how much LC actually fires (your own run-stats
table):

| dataset | loops/scene | why |
|---|---:|---|
| **ScanNet++ 50** | **54.2** | most active — if LC does nothing for the map, it shows here first |
| **ScanNet val-50** | 12.2 | pairs against a 50-scene GT-geometry row |

## 4. ⚠️ Report duplicate count, not only AP

**AP alone cannot answer this question.** A drop in AP is equally consistent with
"objects were not found" and with "one object was split into two", and only the second is
what LC is supposed to fix. Please report, per scene and as a mean:

| column | why |
|---|---|
| ATE (SE3 + Sim3) | ties this run to the existing evidence |
| AP / AP50 / AP25 | the headline |
| **`n_pred`** | over-segmentation moves this even when AP is flat |
| **duplicate instances** | ≥2 predicted clusters matching the same GT instance above the IoU floor — **the quantity the experiment is actually about** |
| align residual RMSE | separates "trajectory drifted" from "perception is worse" |
| loops accepted | confirms the ON arm really had loops to close |

If a duplicate-counter does not exist yet, the cheapest version: for each GT instance,
count predicted clusters with IoU ≥ 0.25 against it; `sum(max(0, k-1))` over GT
instances is the duplicate count. Same evaluator for both arms.

## 5. Nice-to-have

One before/after qualitative on the ScanNet++ scene with the most loops — the same
object as two fragments with LC off, merged with LC on. That is the figure for this
arrow if the effect exists.

## 6. If the effect is null

Report it as null. We will simply not claim `trajectory → map` in the paper; the story
already hedges it ("corrected geometry *can* reduce duplicate instances") and nothing
else depends on it. A measured null on ScanNet++ at 54 loops/scene is a **stronger**
result for us than an unmeasured hedge, because it says the descriptor did the work.

## 7. Reference numbers to land next to (OpenSe3r side, GT depth + GT pose, h2o+geo)

| dataset | AP / AP50 / AP25 |
|---|---|
| ScanNet val-50 (stuff filter) | 42.39 / 67.37 / 82.80 |
| ScanNet++ 50 (top100) | 27.13 / 50.88 / 68.69 |
| ScanNet++ 50 (all) | 19.14 / 37.61 / 55.39 |

Your protocol-B (LC on) numbers for the same cells are 27.12 / 51.86 / 72.68 and
20.64 / 43.44 / 64.68 (top100) — those are the "LC on" arm; only "LC off" is missing.
