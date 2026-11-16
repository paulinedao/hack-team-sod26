**Segment-Level Possession Model – More Than A Hack 2027**

**Objective**

Improve football possession detection by reducing short, potentially incorrect possession switches in the tracking-based predictions.

**Methodology**

We developed a three-stage approach:

1. **Tracking baseline:** Use ATD tracking data to infer possession (Home/Away/Dead ball), based on the team nearest to the ball, followed by temporal smoothing.
2. **Smart Tagging integration:** Incorporate tactical event annotations (e.g., build-up, progression, attacking transitions) to correct possession labels when supported by tracking evidence.
3. **Segment-level refinement:** Identify short, isolated possession segments and merge them into the surrounding possession when there is sufficient evidence.

For example, a sequence `H → A → H` may be corrected to `H → H → H` if:

- The middle possession lasts no longer than 2 seconds.
- Both surrounding possessions belong to the same team and last at least 3 seconds each.
- At least 60% of the observed raw tracking frames within the middle segment support the surrounding team.

The refinement runs for up to three passes and preserves dead-ball (`D`) segments.

**Results**

Across the full match (199,319 frames), the segment-level refinement:

- Merged 4 short possession segments.
- Changed only 109 frames (0.055% of the match).
- Reduced predicted possession spells from 437 to 429.
- Reduced short possessions (under 2 seconds) from 111 to 107.
- Preserved the dead-ball proportion at 29.65%.

**Public evaluation (first 5 minutes, 7,500 frames)**

| Model | Macro F1 | Spell F1 |
|---|---:|---:|
| Smooth100 baseline | 0.5976 | 0.4485 |
| Tracking + Smart Tagging | 0.5982 | 0.4606 |
| **Segment-level model** | **0.6007** | **0.4903** |

The segment model achieved the highest observed spellF1 among our tested approaches, approximately 6.5% higher than the Smart Tagging model.

**Limitations and next steps**

The evaluation covers only 17 ground-truth possession spells, so the improvement is uncertain and may not generalize to the hidden full match.

All four merged segments were corrected from Away to Home, so we should investigate possible directional bias.

Next, we plan to audit these corrections against tracking and tactical annotations, check full-match possession distributions, and validate the final submission format.

**Key takeaway:** Segment-level refinement is a lightweight, interpretable post-processing method that reduces short possession interruptions without requiring model training. Its full-match structural results are encouraging, but hidden-match accuracy remains unverified.

One important point to emphasize to your teammates: this is a rule-based segment refinement model, not a trained machine-learning model. Its main advantage is interpretability—we can explain exactly why each possession segment was changed.
