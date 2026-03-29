"use client";

import { useEffect, useRef } from "react";
import type { Note } from "@/app/page";
import {
  r2g, groupChords, buildVoiceMeasure, type VexClasses,
} from "@/lib/vexHelpers";

// Must match PianoRollViewer's PX_PER_S so barlines align exactly.
const PX_PER_S = 80;

const TREBLE_Y   = 20;
const BASS_Y     = 110;
export const LINEAR_SCORE_H = 230;  // total height of the score strip
export const LINE_SPACING   = 14;   // px between stave lines (controls clef size)
export { TREBLE_Y, BASS_Y };

type Props = {
  notes:      Note[];
  barTimes:   number[];
  timeSigNum: number;
  timeSigDen: number;
  totalWidth: number;
  keySig?:    string;
};

export default function LinearScoreViewer({
  notes, barTimes, timeSigNum, timeSigDen, totalWidth, keySig = "C",
}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || notes.length === 0 || barTimes.length < 2) {
      if (ref.current) ref.current.innerHTML = "";
      return;
    }
    renderLinearScore(ref.current, notes, barTimes, timeSigNum, timeSigDen, totalWidth, keySig)
      .catch(e => console.warn("LinearScoreViewer:", e));
  }, [notes, barTimes, timeSigNum, timeSigDen, totalWidth, keySig]);

  return (
    <div
      ref={ref}
      style={{ height: LINEAR_SCORE_H, width: totalWidth, minWidth: totalWidth }}
      className="bg-white [&_svg]:overflow-visible [&_text]:fill-black [&_path]:stroke-black"
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Async render — single horizontal system, stave widths ∝ bar durations
// ─────────────────────────────────────────────────────────────────────────────

async function renderLinearScore(
  container:  HTMLDivElement,
  rawNotes:   Note[],
  barTimes:   number[],
  timeSigNum: number,
  timeSigDen: number,
  totalWidth: number,
  keySig:     string,
) {
  const { Renderer, Stave, StaveNote, Voice, Formatter, Accidental, Beam } =
    await import("vexflow");

  container.innerHTML = "";

  const notes = [...rawNotes].sort((a, b) => a.start - b.start);
  if (notes.length === 0) return;

  // Extend barTimes by one bar at each end for notes outside the explicit range
  const sorted   = [...barTimes].sort((a, b) => a - b);
  const prevDur  = sorted[1] - sorted[0];
  const lastDur  = sorted[sorted.length - 1] - sorted[sorted.length - 2];
  const allBars  = [sorted[0] - prevDur, ...sorted, sorted[sorted.length - 1] + lastDur];

  const vex: VexClasses = { StaveNote: StaveNote as VexClasses["StaveNote"], Accidental };

  // BPB in quarter-beat units (e.g. 6/8 → 6*(4/8)=3)
  const BPB = timeSigNum * (4 / timeSigDen);

  // Build one Measure { t, b } per bar using allBars intervals
  type SN      = InstanceType<typeof StaveNote>;
  type Measure = { t: SN[]; b: SN[]; barStart: number; barDur: number };

  const measures: Measure[] = [];

  for (let bi = 0; bi < allBars.length - 1; bi++) {
    const barStart = allBars[bi];
    const barEnd   = allBars[bi + 1];
    const barDur   = barEnd - barStart;
    if (barStart < -prevDur * 1.1) continue;           // before visible range
    if (barStart > totalWidth / PX_PER_S + lastDur * 0.1) break; // past end

    // Local beat duration for this bar (may vary if user dragged barlines)
    const beatS = barDur / timeSigNum;

    // Map absolute time → beat index within this bar (0-based)
    const timeToBeat = (t: number) => r2g((t - barStart) / beatS);

    const inBar     = notes.filter(n => n.start >= barStart - 0.02 && n.start < barEnd - 0.02);
    const trebleIn  = inBar.filter(n => n.midi_note >= 60);
    const bassIn    = inBar.filter(n => n.midi_note < 60);

    const trebleChords = groupChords(trebleIn, timeToBeat, beatS);
    const bassChords   = groupChords(bassIn,   timeToBeat, beatS);

    measures.push({
      barStart, barDur,
      t: buildVoiceMeasure(trebleChords, 0, BPB, "treble", vex) as SN[],
      b: buildVoiceMeasure(bassChords,   0, BPB, "bass",   vex) as SN[],
    });
  }

  if (measures.length === 0) return;

  // ── VexFlow layout ────────────────────────────────────────────────────────
  const renderer = new Renderer(container, Renderer.Backends.SVG);
  renderer.resize(totalWidth, LINEAR_SCORE_H);
  const ctx = renderer.getContext();
  // Do NOT set a small font here — VexFlow will use its default glyph scale
  // which correctly sizes clefs relative to the stave line spacing.

  // Width reserved for clef + time sig on the first stave
  const FIRST_OVERHEAD = 160;

  for (let mi = 0; mi < measures.length; mi++) {
    const { barStart, barDur, t, b } = measures[mi];
    const first  = mi === 0;
    const x      = barStart * PX_PER_S;
    const w      = barDur   * PX_PER_S;
    const noteW  = Math.max(4, w - (first ? FIRST_OVERHEAD : 20));

    const staveOpts = { spacing_between_lines_px: LINE_SPACING };
    const ts = new Stave(x, TREBLE_Y, w, staveOpts);
    const bs = new Stave(x, BASS_Y,   w, staveOpts);
    if (first) {
      ts.addClef("treble"); ts.addKeySignature(keySig); ts.addTimeSignature(`${timeSigNum}/${timeSigDen}`);
      bs.addClef("bass");   bs.addKeySignature(keySig); bs.addTimeSignature(`${timeSigNum}/${timeSigDen}`);
    }
    ts.setContext(ctx).draw();
    bs.setContext(ctx).draw();

    if (t.length === 0 && b.length === 0) continue;

    try {
      const tv = new Voice({ num_beats: timeSigNum, beat_value: timeSigDen }).setStrict(false);
      const bv = new Voice({ num_beats: timeSigNum, beat_value: timeSigDen }).setStrict(false);
      tv.addTickables(t);
      bv.addTickables(b);
      new Formatter().joinVoices([tv]).joinVoices([bv]).format([tv, bv], noteW);
      tv.draw(ctx, ts);
      bv.draw(ctx, bs);
      Beam.generateBeams(t.filter(n => !n.isRest())).forEach(beam => beam.setContext(ctx).draw());
      Beam.generateBeams(b.filter(n => !n.isRest())).forEach(beam => beam.setContext(ctx).draw());
    } catch (e) {
      console.warn(`LinearScoreViewer bar ${mi}:`, e);
    }
  }

  const svg = container.querySelector("svg");
  if (svg) svg.style.background = "transparent";
}

// Re-export so PianoRollViewer can import the constant without importing
// the full component eagerly.
export { LINEAR_SCORE_H as SCORE_H };
