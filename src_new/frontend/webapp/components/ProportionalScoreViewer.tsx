"use client";

import { useEffect, useRef, useState } from "react";
import type { Note } from "@/app/page";

type Props = { notes: Note[]; duration: number };

export default function ProportionalScoreViewer({ notes, duration }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");

  useEffect(() => {
    if (!containerRef.current || notes.length === 0) return;

    setError(null);
    setStatus("loading VexFlow…");

    renderScore(containerRef.current, notes)
      .then(() => setStatus("ok"))
      .catch((e: unknown) => {
        console.error("ProportionalScoreViewer error:", e);
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      });
  }, [notes, duration]);

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
        Note values inferred by clustering actual durations · barlines are layout only
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Duration clustering (k-means on log-durations)
// ─────────────────────────────────────────────────────────────────────────────

const STD_VALUES: [number, string][] = [
  [8, "w"], [4, "h"], [2, "q"], [1, "8"], [0.5, "16"], [0.25, "32"],
];

const VEX_BEATS: Record<string, number> = {
  w: 4, h: 2, q: 1, "8": 0.5, "16": 0.25, "32": 0.125,
};

function clusterDurations(notes: Note[]): Map<number, string> {
  const durations = notes.map(n => Math.max(0.01, n.end - n.start));
  const logDurs   = durations.map(d => Math.log2(d));
  const minL = Math.min(...logDurs);
  const maxL = Math.max(...logDurs);

  const k = Math.min(6, new Set(durations.map(d => Math.round(d * 10))).size || 1);
  let centroids = Array.from({ length: k }, (_, i) =>
    k === 1 ? (minL + maxL) / 2 : minL + (i / (k - 1)) * (maxL - minL)
  );

  for (let iter = 0; iter < 30; iter++) {
    const clusters: number[][] = Array.from({ length: k }, () => []);
    for (const l of logDurs) {
      let best = 0, bestD = Infinity;
      centroids.forEach((c, i) => { const d = Math.abs(l - c); if (d < bestD) { bestD = d; best = i; } });
      clusters[best].push(l);
    }
    const next = centroids.map((c, i) =>
      clusters[i].length > 0 ? clusters[i].reduce((a, b) => a + b) / clusters[i].length : c
    );
    if (next.every((v, i) => Math.abs(v - centroids[i]) < 0.001)) break;
    centroids = next;
  }

  const maxC = Math.max(...centroids);
  const centroidToVex = centroids.map(c => {
    const rel = Math.pow(2, c - maxC) * 8;
    let best = STD_VALUES[0], bestD = Infinity;
    for (const sv of STD_VALUES) {
      const d = Math.abs(Math.log2(rel / sv[0]));
      if (d < bestD) { bestD = d; best = sv; }
    }
    return best[1];
  });

  const result = new Map<number, string>();
  notes.forEach((n, ni) => {
    const l = Math.log2(Math.max(0.01, n.end - n.start));
    let best = 0, bestD = Infinity;
    centroids.forEach((c, i) => { const d = Math.abs(l - c); if (d < bestD) { bestD = d; best = i; } });
    result.set(ni, centroidToVex[best]);
  });
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function dedup(notes: Note[]): Note[] {
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const lastSeen = new Map<number, number>();
  return sorted.filter(n => {
    const prev = lastSeen.get(n.midi_note);
    if (prev !== undefined && n.start - prev < 0.06) return false;
    lastSeen.set(n.midi_note, n.start);
    return true;
  });
}

const NOTE_NAMES = ["c","c#","d","d#","e","f","f#","g","g#","a","a#","b"];
function midiToVex(midi: number): { key: string; accidental: string | null } {
  const name   = NOTE_NAMES[midi % 12];
  const octave = Math.floor(midi / 12) - 1;
  return { key: `${name}/${octave}`, accidental: name.includes("#") ? "#" : null };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main render
// ─────────────────────────────────────────────────────────────────────────────

async function renderScore(container: HTMLDivElement, rawNotes: Note[]) {
  // VexFlow 4.0.x exports everything as named exports from the root
  const { Renderer, Stave, StaveNote, Voice, Formatter, Accidental, Beam } =
    await import("vexflow");

  container.innerHTML = "";

  const notes  = dedup(rawNotes);
  if (notes.length === 0) return;

  const durMap = clusterDurations(notes);

  // ── Group simultaneous notes into chord slots ────────────────────────────
  type Slot = { time: number; treble: Note[]; bass: Note[] };
  const slots: Slot[] = [];
  const sorted = [...notes].sort((a, b) => a.start - b.start);

  for (const note of sorted) {
    const last = slots[slots.length - 1];
    if (last && Math.abs(note.start - last.time) < 0.04) {
      (note.midi_note >= 60 ? last.treble : last.bass).push(note);
    } else {
      slots.push({
        time:   note.start,
        treble: note.midi_note >= 60 ? [note] : [],
        bass:   note.midi_note >= 60 ? [] : [note],
      });
    }
  }

  // ── Build StaveNotes ──────────────────────────────────────────────────────
  function makeNote(
    slotNotes: Note[], fallbackDur: string, isRest: boolean, clef: "treble" | "bass"
  ): InstanceType<typeof StaveNote> {
    if (isRest || slotNotes.length === 0) {
      return new StaveNote({
        clef, keys: [clef === "treble" ? "b/4" : "d/3"],
        duration: fallbackDur + "r",
      });
    }
    // Longest note drives the duration of the whole chord
    let bestDur = "32", bestBeats = 0;
    for (const n of slotNotes) {
      const ni  = notes.indexOf(n);
      const dur = durMap.get(ni) ?? "q";
      const b   = VEX_BEATS[dur] ?? 1;
      if (b > bestBeats) { bestBeats = b; bestDur = dur; }
    }
    const keysArr = [...slotNotes]
      .sort((a, b) => a.midi_note - b.midi_note)
      .map(n => midiToVex(n.midi_note).key);

    const sn = new StaveNote({ clef, keys: keysArr, duration: bestDur });
    [...slotNotes]
      .sort((a, b) => a.midi_note - b.midi_note)
      .forEach((n, i) => {
        const { accidental } = midiToVex(n.midi_note);
        if (accidental) sn.addModifier(new Accidental(accidental), i);
      });
    return sn;
  }

  // ── Pack slots into 4/4 measures ─────────────────────────────────────────
  type Measure = { t: InstanceType<typeof StaveNote>[]; b: InstanceType<typeof StaveNote>[] };
  const measures: Measure[] = [];
  let cur: Measure = { t: [], b: [] };
  let tb = 0, bb = 0;
  const BEATS = 4;

  function padStaff(
    arr: InstanceType<typeof StaveNote>[], beats: number, clef: "treble" | "bass"
  ): number {
    while (beats < BEATS) {
      const fill  = (BEATS - beats) >= 2 ? "h" : "q";
      arr.push(makeNote([], fill, true, clef));
      beats += VEX_BEATS[fill];
    }
    return beats;
  }

  for (const slot of slots) {
    const tDur = (() => {
      const n = slot.treble[0]; if (!n) return "q";
      return durMap.get(notes.indexOf(n)) ?? "q";
    })();
    const bDur = (() => {
      const n = slot.bass[0]; if (!n) return "q";
      return durMap.get(notes.indexOf(n)) ?? "q";
    })();
    const tB = VEX_BEATS[tDur] ?? 1;
    const bB = VEX_BEATS[bDur] ?? 1;

    if (tb + tB > BEATS + 0.01 || bb + bB > BEATS + 0.01) {
      padStaff(cur.t, tb, "treble"); padStaff(cur.b, bb, "bass");
      measures.push(cur);
      cur = { t: [], b: [] }; tb = 0; bb = 0;
    }

    cur.t.push(makeNote(slot.treble, tDur, slot.treble.length === 0, "treble")); tb += tB;
    cur.b.push(makeNote(slot.bass,   bDur, slot.bass.length   === 0, "bass"));   bb += bB;
  }
  if (cur.t.length) {
    padStaff(cur.t, tb, "treble"); padStaff(cur.b, bb, "bass");
    measures.push(cur);
  }
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
      const noteW   = STAVE_W - (first ? 62 : 22);

      const ts = new Stave(x, offY + TREBLE_Y, STAVE_W);
      const bs = new Stave(x, offY + BASS_Y,   STAVE_W);
      if (first) { ts.addClef("treble"); bs.addClef("bass"); }
      ts.setContext(ctx).draw();
      bs.setContext(ctx).draw();

      try {
        const tv = new Voice({ num_beats: 4, beat_value: 4 }).setStrict(false);
        const bv = new Voice({ num_beats: 4, beat_value: 4 }).setStrict(false);
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

  // ── Recolour SVG for dark background ─────────────────────────────────────
  const svg = container.querySelector("svg");
  if (svg) {
    svg.style.background = "#16161f";
    svg.querySelectorAll<SVGElement>("path,rect,circle,text,line,polyline").forEach(el => {
      const f = el.getAttribute("fill");
      const s = el.getAttribute("stroke");
      if (f === "black" || f === "#000000" || f === "rgb(0,0,0)") el.setAttribute("fill",   "#e2e8f0");
      if (s === "black" || s === "#000000" || s === "rgb(0,0,0)") el.setAttribute("stroke", "#e2e8f0");
    });
  }
}
