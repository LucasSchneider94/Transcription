"use client";

import { useEffect, useRef } from "react";
import type { Note } from "@/app/page";
import {
  r2g, groupChords, buildVoiceMeasure, splitChordsBetweenVoices, FLAT_KEYS, type VexClasses,
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
  const flatSpelling = FLAT_KEYS.has(keySig);

  // Build one Measure per bar — up to 2 voices per stave for polyphony
  type SN      = InstanceType<typeof StaveNote>;
  type Measure = { t1: SN[]; t2: SN[] | null; b1: SN[]; b2: SN[] | null; barStart: number; barDur: number };

  const measures: Measure[] = [];

  for (let bi = 0; bi < allBars.length - 1; bi++) {
    const barStart = allBars[bi];
    const barEnd   = allBars[bi + 1];
    const barDur   = barEnd - barStart;
    if (barStart < -prevDur * 1.1) continue;
    if (barStart > totalWidth / PX_PER_S + lastDur * 0.1) break;

    const beatS      = barDur / timeSigNum;
    const timeToBeat = (t: number) => r2g((t - barStart) / beatS);

    const inBar    = notes.filter(n => n.start >= barStart - 0.02 && n.start < barEnd - 0.02);
    const trebleIn = inBar.filter(n => n.midi_note >= 60);
    const bassIn   = inBar.filter(n => n.midi_note < 60);

    const [t1c, t2c] = splitChordsBetweenVoices(groupChords(trebleIn, timeToBeat, beatS));
    const [b1c, b2c] = splitChordsBetweenVoices(groupChords(bassIn,   timeToBeat, beatS));

    measures.push({
      barStart, barDur,
      t1: buildVoiceMeasure(t1c, 0, BPB, "treble", vex, flatSpelling) as SN[],
      t2: t2c.length > 0 ? buildVoiceMeasure(t2c, 0, BPB, "treble", vex, flatSpelling) as SN[] : null,
      b1: buildVoiceMeasure(b1c, 0, BPB, "bass",   vex, flatSpelling) as SN[],
      b2: b2c.length > 0 ? buildVoiceMeasure(b2c, 0, BPB, "bass",   vex, flatSpelling) as SN[] : null,
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
    const { barStart, barDur, t1, t2, b1, b2 } = measures[mi];
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

    try {
      type VoiceInst = InstanceType<typeof Voice>;
      const makeVoice = (sns: SN[]): VoiceInst =>
        new Voice({ num_beats: timeSigNum, beat_value: timeSigDen }).setStrict(false)
          .addTickables(sns) as VoiceInst;

      const tv1 = makeVoice(t1);
      const bv1 = makeVoice(b1);
      const tv2 = t2 ? makeVoice(t2) : null;
      const bv2 = b2 ? makeVoice(b2) : null;

      const allVoices = [tv1, bv1, ...(tv2 ? [tv2] : []), ...(bv2 ? [bv2] : [])];

      // Resolve accidentals with key-sig context across all voices in this bar
      (Accidental as unknown as { applyAccidentals: (vs: unknown[], k: string) => void })
        .applyAccidentals(allVoices, keySig);

      const fmt = new Formatter();
      if (tv2) fmt.joinVoices([tv1, tv2]); else fmt.joinVoices([tv1]);
      if (bv2) fmt.joinVoices([bv1, bv2]); else fmt.joinVoices([bv1]);
      fmt.format(allVoices, noteW);

      tv1.draw(ctx, ts);
      tv2?.draw(ctx, ts);
      bv1.draw(ctx, bs);
      bv2?.draw(ctx, bs);

      const beamVoice = (sns: SN[]) =>
        Beam.generateBeams(sns.filter(n => !n.isRest()))
            .forEach(beam => beam.setContext(ctx).draw());
      beamVoice(t1);
      if (t2) beamVoice(t2);
      beamVoice(b1);
      if (b2) beamVoice(b2);
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
