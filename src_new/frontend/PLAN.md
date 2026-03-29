# MVP Plan: AI-Assisted Piano Transcription Editor

## Architecture principle
Inference is compute-heavy → run once on the server, return raw probability arrays (`onset_roll`, `piano_roll`, each T×88 float).  
All downstream decoding, quantization and editing happens client-side in TypeScript — zero re-inference needed.

---

## Phase 1 — Interactive decoding (✅ in progress)

Move note decoding entirely to the browser so parameters update live.

### What to build
- `lib/decode.ts` — TypeScript port of `_decode_notes`:
  - `onset_threshold` (default 0.6)
  - `frame_threshold` (default 0.4)
  - `refractory_ms` — minimum time before same pitch re-triggers (default 50ms)
  - `min_note_ms` — discard notes shorter than this (default 50ms)
  - `onset_gating` toggle — frame must be onset-initiated
  - `frame_smoothing` — median kernel size applied to frame roll before threshold (removes sustain jitter)
- `components/DecodeControls.tsx` — sidebar/panel with sliders + toggles wired to decode params
- `page.tsx` — store raw `onset_roll`/`piano_roll` arrays from API response; re-decode on every param change via `useMemo`

### API change
- Remove `notes` computation from the Python API (or keep as default); ensure `onset_roll` and `piano_roll` are always returned.

---

## Phase 2 — Tempo detection + score quantization

Convert raw note times into a rhythmic grid.

### What to build
- Python `/quantize` endpoint:
  1. Build onset envelope from detected onset times (100fps binary signal)
  2. Autocorrelate → find dominant peak = beat period → BPM
  3. Dynamic programming beat tracking (librosa `beat_track` from onset envelope)
  4. For each note: find nearest grid slot (1/32, 1/16, 1/8, 1/4, 1/2, 1), return quantized duration + offset error
- Frontend: beat grid overlay on piano roll (vertical lines at beat positions)
- Quantization confidence indicator per note (how far from nearest grid slot)

---

## Phase 3 — Score editor (in ProportionalScoreViewer)

Editing happens in the **score view**, not the piano roll, so durations are relative to the beat grid.

### What to build
- Click note → select (highlighted border)
- Keyboard shortcuts on selected note:
  - `↑` / `↓` → pitch ±1 semitone
  - `←` / `→` → shift note ±1 grid slot
  - `[` / `]` → halve / double duration (1/32 ↔ 1/16 ↔ 1/8 ↔ 1/4 ↔ 1/2 ↔ 1)
  - `Delete` → remove note
- "Trim sustain" button: shorten each note to where the frame probability drops below threshold (fixes overheld notes)
- Undo/redo stack (array of note-list snapshots, Ctrl+Z / Ctrl+Y)

---

## Phase 4 — Export

- `/export/midi` endpoint: edited note list → MIDI file download
- `/export/musicxml` endpoint: quantized + edited notes → MusicXML → PDF via LilyPond or MuseScore CLI

---

## Color / style reference
- Accent purple: `#7c6af7` (`accent`) / `#a89df9` (`accent-light`) — defined in `tailwind.config.ts`
- Background: `#0f0f13`, Surface: `#1a1a24`, Border: `#2a2a3a`
