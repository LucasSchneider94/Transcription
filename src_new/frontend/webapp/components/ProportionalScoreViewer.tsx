"use client";

import { useEffect, useRef, useState } from "react";
import type { Note } from "@/app/page";

type Props = { notes: Note[]; duration: number; bpm: number; beatOffset?: number; timeSigNum?: number; timeSigDen?: number };

export default function ProportionalScoreViewer({ notes, duration, bpm, beatOffset = 0, timeSigNum = 4, timeSigDen = 4 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");

  useEffect(() => {
    if (!containerRef.current || notes.length === 0) return;

    setError(null);
    setStatus("loading VexFlow…");

    renderScore(containerRef.current, notes, bpm, beatOffset, timeSigNum, timeSigDen)
      .then(() => setStatus("ok"))
      .catch((e: unknown) => {
        console.error("ProportionalScoreViewer error:", e);
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      });
  }, [notes, duration, bpm, beatOffset, timeSigNum, timeSigDen]);

  return (
    <div className="space-y-2">
      {status === "loading VexFlow…" && (
        <p className="text-xs text-muted animate-pulse">Loading score…</p>
      )}
      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 rounded-xl px-4 py-3 text-sm font-mono whitespace-pre-wrap">
          {error}
        </div>
      )}
      <div
        ref={containerRef}
        className="rounded-xl bg-white overflow-x-hidden overflow-y-auto max-h-[80vh] p-2 [&_svg]:text-black [&_text]:fill-black [&_path]:stroke-black"
      />
      <p className="text-xs text-muted">
        {timeSigNum}/{timeSigDen} · {bpm} BPM · offset {(beatOffset * 1000).toFixed(0)} ms · {notes.length} notes
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const NOTE_NAMES = ["c","c#","d","d#","e","f","f#","g","g#","a","a#","b"];
function midiToVex(midi: number): { key: string; accidental: string | null } {
  const name   = NOTE_NAMES[midi % 12];
  const octave = Math.floor(midi / 12) - 1;
  return { key: `${name}/${octave}`, accidental: name.includes("#") ? "#" : null };
}

// ─────────────────────────────────────────────────────────────────────────────
// BPM-based duration helpers
// ─────────────────────────────────────────────────────────────────────────────

const BEAT_DURS: [number, string][] = [
  [4, "w"], [2, "h"], [1, "q"], [0.5, "8"], [0.25, "16"], [0.125, "32"],
];

/** Round to nearest 32nd-note grid (0.125 beats). */
function r2g(beats: number): number {
  return Math.round(beats * 8) / 8;
}

function beatDurToVex(beats: number): string {
  let best = "q", bestDist = Infinity;
  for (const [val, name] of BEAT_DURS) {
    const d = Math.abs(Math.log2(Math.max(0.001, beats) / val));
    if (d < bestDist) { bestDist = d; best = name; }
  }
  return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main render
// ─────────────────────────────────────────────────────────────────────────────

async function renderScore(
  container: HTMLDivElement, rawNotes: Note[], bpm: number,
  beatOffset: number, timeSigNum: number, timeSigDen: number,
) {
  const { Renderer, Stave, StaveNote, Voice, Formatter, Accidental, Beam } =
    await import("vexflow");

  container.innerHTML = "";

  // No filtering here — the score renders exactly what it receives.
  const notes = [...rawNotes].sort((a, b) => a.start - b.start);
  if (notes.length === 0) return;

  // Bar length in quarter-note beats: e.g.  6/8 → BPB = 6*(4/8) = 3.0
  const BPB = timeSigNum * (4 / timeSigDen);

  type SN      = InstanceType<typeof StaveNote>;
  type Measure = { t: SN[]; b: SN[] };

  // ── BPM mode: independent per-voice timeline → no cross-voice rest drift ──
  function buildMeasuresBPM(tempo: number): Measure[] {
    const beatS  = 60 / tempo;

    // Convert absolute time → beat position, accounting for the beat offset
    // so the score grid aligns with the same grid shown in the piano roll.
    function timeToBeat(t: number): number {
      return r2g((t - beatOffset) / beatS);
    }

    // Find the beat position of the first note and floor to the nearest bar
    // so the score starts at bar 1 beat 1 rather than leaving dead rests.
    const firstBeat = notes.length
      ? r2g(Math.min(...notes.map(n => n.start - beatOffset)) / beatS)
      : 0;
    const barOrigin = Math.floor(firstBeat / BPB) * BPB;  // floor to bar boundary

    interface Chord { startBeat: number; durBeats: number; midiNotes: number[] }

    function groupChords(voiceNotes: Note[]): Chord[] {
      const sorted = [...voiceNotes].sort((a, b) => a.start - b.start);
      const chords: Chord[] = [];
      for (const n of sorted) {
        const sb = timeToBeat(n.start) - barOrigin;
        const db = Math.max(0.125, r2g((n.end - n.start) / beatS));
        const last = chords[chords.length - 1];
        // group notes within one 32nd note of each other as a chord
        if (last && Math.abs(sb - last.startBeat) < 0.13) {
          last.midiNotes.push(n.midi_note);
          last.durBeats = Math.max(last.durBeats, db);
        } else {
          chords.push({ startBeat: sb, durBeats: db, midiNotes: [n.midi_note] });
        }
      }
      return chords;
    }

    function makeRestsBeats(beats: number, restKey: string, clef: string): SN[] {
      const out: SN[] = [];
      let rem = r2g(beats);
      for (const [val, name] of BEAT_DURS) {
        while (rem >= val - 0.05) {
          out.push(new StaveNote({ clef, keys: [restKey], duration: name + "r" }));
          rem = r2g(rem - val);
        }
      }
      return out;
    }

    function buildVoiceMeasure(
      chords: Chord[], measureStart: number, clef: "treble" | "bass",
    ): SN[] {
      const mEnd    = measureStart + BPB;
      const restKey = clef === "treble" ? "b/4" : "d/3";
      const inM    = chords.filter(c => c.startBeat >= measureStart && c.startBeat < mEnd);
      const out: SN[] = [];
      let pos = measureStart;

      for (const chord of inM) {
        const gap = r2g(chord.startBeat - pos);
        if (gap >= 0.125) {
          out.push(...makeRestsBeats(gap, restKey, clef));
          pos = chord.startBeat;
        }
        const dur    = r2g(Math.min(chord.durBeats, mEnd - chord.startBeat));
        const vexDur = beatDurToVex(Math.max(0.125, dur));
        const sorted = [...chord.midiNotes].sort((a, b) => a - b);
        const keys   = sorted.map(m => midiToVex(m).key);
        const sn     = new StaveNote({ clef, keys, duration: vexDur });
        sorted.forEach((m, i) => {
          const { accidental } = midiToVex(m);
          if (accidental) sn.addModifier(new Accidental(accidental), i);
        });
        out.push(sn);
        pos = r2g(chord.startBeat + dur);
      }

      const tail = r2g(mEnd - pos);
      if (tail >= 0.125) out.push(...makeRestsBeats(tail, restKey, clef));
      return out;
    }

    const trebleChords = groupChords(notes.filter(n => n.midi_note >= 60));
    const bassChords   = groupChords(notes.filter(n => n.midi_note < 60));

    const lastBeat = Math.max(
      trebleChords.length ? trebleChords[trebleChords.length - 1].startBeat + trebleChords[trebleChords.length - 1].durBeats : 0,
      bassChords.length   ? bassChords  [bassChords.length   - 1].startBeat + bassChords  [bassChords.length   - 1].durBeats : 0,
      BPB,
    );
    const nMeasures = Math.ceil(lastBeat / BPB);
    return Array.from({ length: nMeasures }, (_, mi) => ({
      t: buildVoiceMeasure(trebleChords, mi * BPB, "treble"),
      b: buildVoiceMeasure(bassChords,   mi * BPB, "bass"),
    }));
  }

  // Always use the BPM path — score and piano roll share the same grid.
  const measures = buildMeasuresBPM(bpm);
  if (measures.length === 0) return;

  // ── Lay out systems ───────────────────────────────────────────────────────
  const SYSTEM_W  = Math.max(600, (container.clientWidth || 900) - 16);
  const STAVE_MIN = 200;
  const M_PER_SYS = Math.max(1, Math.floor(SYSTEM_W / STAVE_MIN));
  const STAVE_W   = Math.floor(SYSTEM_W / M_PER_SYS);
  const TREBLE_Y  = 24;
  const BASS_Y    = 110;
  const SYS_H     = 200;
  const N_SYS     = Math.ceil(measures.length / M_PER_SYS);
  const TOTAL_H   = N_SYS * SYS_H + 24;

  const renderer = new Renderer(container, Renderer.Backends.SVG);
  renderer.resize(SYSTEM_W, TOTAL_H);
  const ctx = renderer.getContext();
  ctx.setFont("Arial", 10);

  for (let sys = 0; sys < N_SYS; sys++) {
    const mStart  = sys * M_PER_SYS;
    const mEnd    = Math.min(mStart + M_PER_SYS, measures.length);
    const offY    = sys * SYS_H;

    for (let mi = mStart; mi < mEnd; mi++) {
      const x       = (mi - mStart) * STAVE_W;
      const first   = mi === mStart;
      const m       = measures[mi];
      const noteW   = STAVE_W - (first ? 90 : 22);

      const ts = new Stave(x, offY + TREBLE_Y, STAVE_W);
      const bs = new Stave(x, offY + BASS_Y,   STAVE_W);
      if (first) {
        ts.addClef("treble"); ts.addTimeSignature(`${timeSigNum}/${timeSigDen}`);
        bs.addClef("bass");   bs.addTimeSignature(`${timeSigNum}/${timeSigDen}`);
      }
      ts.setContext(ctx).draw();
      bs.setContext(ctx).draw();

      try {
        const tv = new Voice({ num_beats: timeSigNum, beat_value: timeSigDen }).setStrict(false);
        const bv = new Voice({ num_beats: timeSigNum, beat_value: timeSigDen }).setStrict(false);
        tv.addTickables(m.t);
        bv.addTickables(m.b);
        new Formatter().joinVoices([tv]).joinVoices([bv]).format([tv, bv], noteW);
        tv.draw(ctx, ts);
        bv.draw(ctx, bs);
        Beam.generateBeams(m.t.filter(n => !n.isRest())).forEach(b => b.setContext(ctx).draw());
        Beam.generateBeams(m.b.filter(n => !n.isRest())).forEach(b => b.setContext(ctx).draw());
      } catch (e) {
        // swallow per-measure VexFlow layout errors
        console.warn("VexFlow measure error:", e);
      }
    }
  }

  // ── Ensure SVG is transparent so the white container bg shows through ────
  const svg = container.querySelector("svg");
  if (svg) {
    svg.style.background = "transparent";
  }
}
