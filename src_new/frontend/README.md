# Piano Transcription – Frontend

Two processes need to run simultaneously: the **FastAPI** inference backend and the **Next.js** frontend.

---

## 1 · Python backend (FastAPI)

```bash
cd src_new/frontend/api
source ../../../myenv/bin/activate   # use the repo's existing venv

# install deps (first time only)
pip install -r requirements.txt

# point to your checkpoint (defaults to training_run_056/best_model.pt)
export CHECKPOINT_PATH=../../../src_new/training_run_056/best_model.pt

uvicorn main:app --reload --port 8000
```

Tune detection thresholds without restarting:
```bash
export ONSET_THRESH=0.4   # default 0.5
export FRAME_THRESH=0.25  # default 0.3
```

---

## 2 · Next.js frontend

```bash
cd src_new/frontend/webapp
npm install   # first time only
npm run dev
```

Open `http://localhost:3000`.

---

## Workflow

1. **Drop** a `.wav` / `.mp3` / `.flac` file onto the upload zone.
2. Drag the **time-range** handles to select a sub-region (defaults to full file).
3. Click **Analyse** — audio is sent to FastAPI, model runs, notes are returned.
4. **Piano Roll** view — scrollable canvas, exact model output, 88-key range.
5. **Score** view — proportional score: noteheads on a grand staff, time is linear
   left→right, no barlines, faint tail shows note duration. Human adds barlines later.

---

## Architecture

```
Browser (Next.js :3000)
    │  FormData (file, start_time, end_time)
    ▼
FastAPI (:8000)  /analyze     ← Next.js rewrites /api/* → localhost:8000/*
    │
    ├── inference.py
    │     ├── librosa mel-spectrogram  (48 kHz, 352 mels, hop=480)
    │     ├── PianoTranscriptionModel  (CNN + Transformer, src_new/model.py)
    │     │     outputs: onset (T,88)  frame (T,88)  duration (T,88)
    │     └── _predictions_to_notes()
    │           per-pitch refractory period → clean note list
    │
    └── JSON  { duration, fps, piano_roll, onset_roll, notes[] }

Browser renders:
    ├── PianoRollViewer       — Canvas, raw frame activations
    └── ProportionalScoreViewer — Canvas, grand staff, proportional time,
                                  noteheads + stems + accidentals, no barlines
```

---

## Score view design rationale

Rhythm quantisation from a live recording is an unsolved research problem —
even professional notation software (Sibelius, Dorico) requires heavy manual
cleanup when importing live MIDI. Rather than guess a tempo and force notes
onto a beat grid (which always looks wrong), the score view is **proportional**:

- **x-axis = real time** (pixels per second, same as piano roll)
- **y-axis = chromatic pitch** mapped onto treble + bass staves
- **No barlines** — the human adds these manually once they know the tempo
- **Duration tails** — a faint horizontal line extends from each notehead to
  show how long the note rings, exactly as in the piano roll

This is standard practice in contemporary music notation (Ligeti, Feldman,
Xenakis) and is the most honest representation of what the model actually knows.

---

## Recording Spotify (or any system audio) on macOS

**BlackHole** is a free virtual audio loopback driver.
`brew install blackhole-2ch` installs it as a system audio device.

### Why your WAV might be silent even though ffmpeg ran fine

ffmpeg reported `3064 KiB` for 18 s — that's correct PCM size, so ffmpeg
**did** capture something. Silent recordings almost always mean one of:

1. Spotify (and most sandboxed apps) **does not follow the system output** —
   it has its own audio session. You must force it through BlackHole.
2. The Multi-Output Device was created but **not set as the active output
   before Spotify started playing** — Spotify locks its audio session on launch.
3. On **macOS Sonoma / Sequoia**, the privacy prompt for microphone/audio
   capture must be accepted for the terminal app (iTerm2, Terminal.app).

---

### Setup that actually works (tested macOS Ventura–Sequoia)

#### Step 1 — Create the Multi-Output Device (one-time)

1. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
2. Bottom-left **+** → **Create Multi-Output Device**
3. Tick **both**:
   - ✅ BlackHole 2ch
   - ✅ Your speakers / headphones (e.g. "MacBook Pro Speakers")
4. Set **BlackHole 2ch** as the **clock source** (right-click it in the list)
5. Rename it "Loopback" for clarity (double-click the name)

#### Step 2 — Route Spotify through it

This is the step most guides miss:

1. **Quit Spotify completely** (Cmd+Q, not just close window)
2. **System Settings → Sound → Output** → select **Loopback**
   *(you will still hear audio through your speakers)*
3. **Relaunch Spotify** — it must start *after* the output is switched
4. Play a track and confirm the level meters in Audio MIDI Setup are moving
   on the BlackHole 2ch device

#### Step 3 — Record with ffmpeg

```bash
# Stereo, 48 kHz (matches the model exactly), mono mix
ffmpeg -f avfoundation \
       -i ":BlackHole 2ch" \
       -ar 48000 \
       -ac 1 \
       recording.wav
# Press q to stop
```

`-ac 1` converts to mono — your model expects mono input.

#### Step 4 — Restore your audio output

After recording, go back to **System Settings → Sound → Output** and
reselect your normal speakers/headphones.

---

### Verify the recording has signal

```bash
# Should print a non-zero RMS value (e.g. -18 dB, not -91 dB)
ffmpeg -i recording.wav -af "astats" -f null - 2>&1 | grep RMS
```

If RMS is `-91 dB` (digital silence), the routing wasn't active.
If RMS is e.g. `-18 dB`, the file is good — drag it into the UI.

---

### Alternative: Loopback app (Rogue Amoeba, ~$99)

Loopback does the same thing with a GUI and handles sandboxed apps like
Spotify more reliably. It also lets you record per-app without touching
system output.  No terminal needed.

### Alternative: SoundSource (Rogue Amoeba, ~$39)

Lets you redirect individual apps to different outputs, so you can send
Spotify → BlackHole while keeping everything else on speakers.

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_PATH` | `src_new/training_run_056/best_model.pt` | Path to model `.pth` file |
| `ONSET_THRESH` | `0.5` | Minimum onset confidence to trigger a note |
| `FRAME_THRESH` | `0.3` | Frame confidence below which a note ends |
| `DEVICE` | auto | `mps` / `cuda` / `cpu` — detected automatically |
