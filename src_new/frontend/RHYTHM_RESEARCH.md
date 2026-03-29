# Rhythm Quantization & Score Transcription — Research Notes

## The core problem

Converting a **human performance** (note onsets in seconds) to **notated rhythm** (rational positions
relative to a beat grid) is called **Automatic Rhythm Transcription (ART)** or
**performance-to-score quantization**. It is genuinely unsolved at human level:

- Even a "steady" human tempo has ±5–15 % local variation per beat (rubato)
- Tuplets (3-lets, 5-lets, 7-lets) are locally ambiguous from timing alone
- The same sound can be written multiple equivalent ways (tied 8ths vs. quarter, etc.)

---

## State-of-the-art algorithms

### Beat tracking (replacing our autocorrelation BPM)

| Method | Key idea | Practical |
|---|---|---|
| **madmom DBN beat tracker** (Böck et al. 2016) | Dynamic Bayesian Network; produces per-beat timestamps that *follow* tempo changes | ✅ `pip install madmom`, ~5 lines of Python |
| **BeatNet** (Heydari et al. 2021) | Online neural beat tracker; better on complex / live music | ✅ pip installable |

The critical upgrade over our autocorrelation: instead of a single global BPM we get
`[t0, t1, t2, …]` — actual beat timestamps that flex with the performer.
Each inter-beat interval becomes a local time unit so rubato is handled cleanly.

### Rhythm quantization once we have beats

| Method | Key idea |
|---|---|
| **Cemgil et al. 2000** | Bayesian HMM; models each onset as Gaussian around its true beat position |
| **Nakamura et al. 2015–2017** | Full stochastic model of performance deviations; best published results |
| **partitura** (TU Graz, Cancino-Chacón) | Python library with `quantize_performance()`; outputs MusicXML directly. Probably the best practical drop-in. |

### End-to-end learnable approaches

| Method | Notes |
|---|---|
| **MT3** (Google, 2021) | T5 Transformer trained end-to-end audio → token sequence; produces near-score MIDI directly |
| **Seq2seq on MIDI pairs** | Train on (performance MIDI, score MIDI) aligned pairs; ASAP dataset (Foscarin et al.) has ~220 pairs; MAESTRO has the audio |

Tuplet detection specifically: can be done heuristically post-quantization — if N onsets
fall inside one beat interval with residual < threshold, label them an N-tuplet.

---

## Recommended implementation path

1. **Add madmom to the Python API** — replace BPM autocorrelation with DBN beat tracker.
   Return `beat_times: float[]` alongside the existing `onset_roll`/`piano_roll`.

2. **Rework client-side quantization** — instead of `snap to nearest (t - offset) / gridS`,
   snap each note to nearest beat fraction *within its containing inter-beat interval*.
   Local grid = `beat[i]` to `beat[i+1]`, subdivided by the chosen subdivision.

3. **Tuplet detection** — after snapping, check each beat interval: if N notes land inside
   with residual < 0.08 beats, label the interval as an N-tuplet (N ∈ {3, 5, 6, 7}).

4. **partitura** as an alternative to step 2+3 entirely — pass note list + beat_times to
   `partitura.musicanalysis.quantize` and get rational positions back.

---

## Human-in-the-loop approach (Phase 3 of original plan)

The most pragmatic path for complex music:

- Use the adaptive beat grid (step 1–2 above) to get 80–90 % right automatically
- Phase 3 editor lets the user:
  - **Drag note left/right** to adjust its beat position within a measure
  - **Right-click → relabel** rhythm value (quarter, 8th, triplet-8th, …)
  - **Per-measure override** for the meter (3/4, 5/4, etc.)
  - **Trim sustain** — shorten note end to first frame dropout point

This is a focused, composable UI: most notes will be right, only outliers need manual
correction, and each action is reversible (undo stack).

---

## References

- Böck et al. *Joint Beat and Downbeat Tracking with Recurrent Neural Networks*, ISMIR 2016
- Nakamura et al. *Merged-Output HMM for Score-Performance Matching*, ISMIR 2015
- Cancino-Chacón et al. *partitura: A Python Package for Symbolic Music Processing*, ISMIR 2022
- Foscarin et al. *ASAP: A Dataset of Aligned Scores and Performances*, ISMIR 2020
- Gardner et al. *MT3: Multi-Task Multitrack Music Transcription*, ICLR 2022
