# E10 results — VLM saturation bench (2026-10-11)

Sources are files in the private research workspace (`dataset/`), given as `file:line`. Ground truth is Dhruva's
(from the gold review notes); on easy cases it is Astra's earlier label that Dhruva confirmed. The bench cases
themselves (labels) are not published; the code reads them from a private `cases.json`.

## Positions
Source: `reviews/autolabel-v1/REVIEW.md:334-359`; `repo: src/vlm_bench.py`.
17 gold position cases (8 hard). Each model answered blind; the Claude runs are fresh Claude Opus 5.5 agents that could
not see the notes. A case is fully right when position, detail and top athlete match (where scored).

| run | fully right | hard |
|---|---|---|
| Astra high, single frame | 13/17 | 5/8 |
| Astra high, 4 s strip (outlines / tags only) | 13 / 12 | 5/8 |
| Astra high, frame + verify pass | 14/17 | 5/8 |
| Astra extra-high, frame / strip | 13 / 14 | 5/8 |
| Claude Opus, frame / + verify / strip / tags | 9 / 8 / 6 / 5 | 4 / 3 / 2 / 2 |

Production used Astra at medium effort with no verify pass; the gold review measured it at 11/20 fully right on
(different) position items.

- Astra saturates at 13-14/17. The same three cases fail under every setting: a closed guard being opened, a leg-drag
  pin almost passed, a takedown attempt vs turtle. Dhruva's own notes call these borderline or provisional ("we don't
  tell positions from single frames").
- Claude Opus is clearly behind on fine detail: it confuses closed and open guard, 50/50 and ashi, body triangle and
  guard, and answers with low confidence. Its verify pass changed one answer, for the worse. Temporal strips hurt it.
- Caveats: some prompt definitions were written after reading Dhruva's notes, and the verify rule "knee crossing =
  side control" after viewing one case, so the hard-case gains are optimistic. On easy cases the ground truth is
  Astra's own earlier label that Dhruva confirmed. Needs a held-out check on fresh gold items.
- Astra usage for the whole bench: about 2% of the weekly window.

## Events
Source: `reviews/autolabel-v1/REVIEW.md:361-385`; `repo: src/vlm_bench_events.py`.
16 gold event items (4 hard: the original label was wrong), ground truth from Dhruva's verdicts and notes. A case is
right when a listed event has the right type, outcome and actor.

| run | type found | + outcome | fully right | hard |
|---|---|---|---|---|
| Astra high, whole interval at 2 fps | 14/16 | 13/16 | **13/16** | 2/4 |
| Astra high, event +-4 s at 4 fps | 14/16 | 9/16 | 9/16 | 1/4 |
| Claude Opus (blind), event +-4 s at 4 fps | 12/16 | 7/16 | 7/16 | 1/4 |

The production labels scored 12/18 in the gold review.

- Longer context beats denser frames: the focused 4 fps window lost outcomes because it cannot see what came before
  or after.
- The whole-interval score is a little lenient: it lists about three events per case, so finding the type is easier.
- Failures common to every run: an arm-triangle escape read as ongoing or failed; a finished guillotine where, checked
  by eye, the tap is not visible at 4 fps (only a guard guillotine, a sudden release and both lying flat). Dhruva infers
  the finish from context; the models do not.
- Conclusion: as with positions, Astra is near saturation with the improved prompt. What remains needs inference
  beyond the pixels (finishes, round ends) or longer context. Audio (taps, the buzzer, the referee) would be the next
  independent signal.
