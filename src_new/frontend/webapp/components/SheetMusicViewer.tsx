"use client";

import { useEffect, useRef } from "react";
import type { Note } from "@/app/page";

type Props = { notes: Note[] };

export default function SheetMusicViewer({ notes }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<unknown>(null);

  useEffect(() => {
    if (!containerRef.current || notes.length === 0) return;

    import("opensheetmusicdisplay").then(({ OpenSheetMusicDisplay }) => {
      const musicXml = notesToMusicXML(notes);

      if (osmdRef.current) {
        const osmd = osmdRef.current as InstanceType<typeof OpenSheetMusicDisplay>;
        osmd.load(musicXml).then(() => osmd.render());
      } else {
        const osmd = new OpenSheetMusicDisplay(containerRef.current!, {
          autoResize: true,
          backend: "svg",
          darkMode: true,
          drawTitle: false,
        });
        osmdRef.current = osmd;
        osmd.load(musicXml).then(() => osmd.render());
      }
    });
  }, [notes]);

  return (
    <div className="space-y-2">
      <div
        ref={containerRef}
        className="bg-[#16161f] rounded-xl p-4 min-h-[200px] overflow-x-auto"
      />
      <p className="text-xs text-muted">
        Sheet music quantised to a 16th-note grid at 120 bpm · ties used for long notes · review before exporting.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const BPM           = 120;
const BEAT_S        = 60 / BPM;           // 0.5 s per quarter note
const BEATS_PER_BAR = 4;
const BAR_S         = BEATS_PER_BAR * BEAT_S;
// Grid: 4 subdivisions per quarter = 16th-note grid
// Coarser than before so durations snap more naturally
const GRID_DIVS     = 4;                  // 16th notes per quarter
const DIVISIONS     = GRID_DIVS;          // MusicXML <divisions>
const GRID_S        = BEAT_S / GRID_DIVS; // seconds per 16th note  (0.125 s @ 120 bpm)
const BAR_GRID      = BEATS_PER_BAR * GRID_DIVS; // 16 grid units per bar
const TREBLE_SPLIT  = 60;                 // Middle C — below → bass staff

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface QNote {
  midi:      number;
  gridStart: number;  // grid units from t = 0
  gridEnd:   number;  // exclusive
  staff:     1 | 2;
}

// A single notehead event as it will appear in the XML stream
interface XmlNote {
  gridPos:   number;  // absolute grid position
  xmlDur:    number;  // duration in grid units
  type:      string;  // "whole" | "half" | "quarter" | "eighth" | "16th"
  dots:      number;
  midi:      number;
  tieStart:  boolean;
  tieStop:   boolean;
  isRest:    boolean;
  staff:     1 | 2;
  voice:     number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function secToGrid(sec: number): number {
  return Math.round(sec / GRID_S);
}

function midiToStep(midi: number): { step: string; octave: number; alter: number } {
  const STEPS  = ["C","C","D","D","E","F","F","G","G","A","A","B"];
  const ALTERS = [ 0,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0];
  const s = midi % 12;
  return { step: STEPS[s], alter: ALTERS[s], octave: Math.floor(midi / 12) - 1 };
}

/**
 * Canonical note values in grid units (16th-note grid, GRID_DIVS=4 per quarter).
 * Whole=16, Half=8, Quarter=4, Eighth=2, 16th=1
 * Dotted versions: dotted-half=12, dotted-quarter=6, dotted-eighth=3
 */
const NOTE_TABLE: [number, string, number][] = [
  // [gridUnits, type, dots]
  [16, "whole",   0],
  [12, "half",    1],
  [ 8, "half",    0],
  [ 6, "quarter", 1],
  [ 4, "quarter", 0],
  [ 3, "eighth",  1],
  [ 2, "eighth",  0],
  [ 1, "16th",    0],
];

/**
 * Decompose `gridUnits` into a sequence of standard note values using ties.
 * E.g. 10 → [8 (half), 2 (eighth)]  →  half tied to eighth
 */
function decomposeDuration(gridUnits: number): Array<{ xmlDur: number; type: string; dots: number }> {
  const result: Array<{ xmlDur: number; type: string; dots: number }> = [];
  let remaining = Math.max(1, gridUnits);

  while (remaining > 0) {
    // Pick the largest value that fits
    let chosen = NOTE_TABLE[NOTE_TABLE.length - 1]; // fallback: 16th
    for (const entry of NOTE_TABLE) {
      if (entry[0] <= remaining) { chosen = entry; break; }
    }
    result.push({ xmlDur: chosen[0], type: chosen[1], dots: chosen[2] });
    remaining -= chosen[0];
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the flat per-staff event stream
// ─────────────────────────────────────────────────────────────────────────────

/**
 * For one staff, produce a flat list of XmlNote events (including rests and
 * tied note segments) sorted by gridPos.
 *
 * Strategy: treat each pitch as an independent lane (like the piano roll does).
 * We merge all pitch lanes into a single voice by collecting all note-on/off
 * events, then for each time slot emit chords where multiple pitches are active
 * at the same onset, and fill gaps with rests.
 *
 * To handle overlapping notes at different pitches we use a simple greedy
 * approach: group notes that share the same onset into chords, then advance
 * the cursor to the next onset. This faithfully mirrors the piano roll.
 */
function buildStaffEvents(notes: QNote[], staffId: 1 | 2, totalGrid: number): XmlNote[] {
  const staffNotes = notes.filter(n => n.staff === staffId);
  if (staffNotes.length === 0) return [];

  // Group by onset position
  const onsetMap = new Map<number, QNote[]>();
  for (const n of staffNotes) {
    if (!onsetMap.has(n.gridStart)) onsetMap.set(n.gridStart, []);
    onsetMap.get(n.gridStart)!.push(n);
  }

  const onsets = [...onsetMap.keys()].sort((a, b) => a - b);
  const events: XmlNote[] = [];
  let cursor = 0;

  for (const onset of onsets) {
    // Fill gap with rest(s)
    if (onset > cursor) {
      const restSegs = decomposeDuration(onset - cursor);
      let restCursor = cursor;
      for (const seg of restSegs) {
        events.push({
          gridPos: restCursor, xmlDur: seg.xmlDur, type: seg.type, dots: seg.dots,
          midi: -1, tieStart: false, tieStop: false, isRest: true,
          staff: staffId, voice: 1,
        });
        restCursor += seg.xmlDur;
      }
    }

    const notesHere = onsetMap.get(onset)!.sort((a, b) => a.midi - b.midi);

    // Use the longest note at this onset to determine chord duration
    const maxDur = Math.max(...notesHere.map(n => n.gridEnd - n.gridStart));

    // Decompose that duration into tied segments
    const segs = decomposeDuration(maxDur);

    segs.forEach((seg, segIdx) => {
      const segPos    = onset + segs.slice(0, segIdx).reduce((s, x) => s + x.xmlDur, 0);
      const isTieStart = segs.length > 1 && segIdx < segs.length - 1;
      const isTieStop  = segIdx > 0;

      notesHere.forEach((n, noteIdx) => {
        events.push({
          gridPos:  segPos,
          xmlDur:   seg.xmlDur,
          type:     seg.type,
          dots:     seg.dots,
          midi:     n.midi,
          tieStart: isTieStart,
          tieStop:  isTieStop,
          isRest:   false,
          staff:    staffId,
          voice:    1,
          // chord: all notes after the first at the same segPos are chords
          // — handled during XML serialisation by checking noteIdx
        } as XmlNote & { noteIdx: number });
        (events[events.length - 1] as unknown as Record<string, number>)["noteIdx"] = noteIdx;
      });
    });

    cursor = onset + maxDur;
  }

  // Final rest to fill bar
  if (cursor < totalGrid) {
    const restSegs = decomposeDuration(totalGrid - cursor);
    let rc = cursor;
    for (const seg of restSegs) {
      events.push({
        gridPos: rc, xmlDur: seg.xmlDur, type: seg.type, dots: seg.dots,
        midi: -1, tieStart: false, tieStop: false, isRest: true,
        staff: staffId, voice: 1,
      });
      rc += seg.xmlDur;
    }
  }

  return events;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialise events to MusicXML note elements
// ─────────────────────────────────────────────────────────────────────────────

function eventToXml(ev: XmlNote, noteIdx: number): string {
  if (ev.isRest) {
    return `\n        <note><rest/><duration>${ev.xmlDur}</duration><type>${ev.type}</type>${ev.dots ? "<dot/>" : ""}<staff>${ev.staff}</staff></note>`;
  }

  const { step, alter, octave } = midiToStep(ev.midi);
  const chordTag  = noteIdx > 0 ? "<chord/>" : "";
  const alterTag  = alter !== 0 ? `<alter>${alter}</alter>` : "";
  const dotTag    = ev.dots ? "<dot/>" : "";
  const tieStartTag = ev.tieStart ? `<tie type="start"/>` : "";
  const tieStopTag  = ev.tieStop  ? `<tie type="stop"/>` : "";
  const notationStart = ev.tieStart ? `<tied type="start"/>` : "";
  const notationStop  = ev.tieStop  ? `<tied type="stop"/>` : "";
  const notationsBlock = (ev.tieStart || ev.tieStop)
    ? `<notations>${notationStop}${notationStart}</notations>` : "";

  return `
        <note>
          ${chordTag}
          <pitch><step>${step}</step>${alterTag}<octave>${octave}</octave></pitch>
          <duration>${ev.xmlDur}</duration>
          ${tieStopTag}${tieStartTag}
          <type>${ev.type}</type>${dotTag}
          ${notationsBlock}
          <staff>${ev.staff}</staff>
        </note>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

function notesToMusicXML(notes: Note[]): string {
  if (notes.length === 0) return "";

  // 1. Quantise to 16th-note grid, deduplicate
  const seen = new Set<string>();
  const qnotes: QNote[] = [];
  for (const n of notes) {
    const gs = secToGrid(n.start);
    const ge = Math.max(gs + 1, secToGrid(n.end));
    const key = `${n.midi_note}@${gs}`;
    if (seen.has(key)) continue;
    seen.add(key);
    qnotes.push({
      midi:      n.midi_note,
      gridStart: gs,
      gridEnd:   ge,
      staff:     n.midi_note >= TREBLE_SPLIT ? 1 : 2,
    });
  }

  if (qnotes.length === 0) return "";

  const totalGrid = Math.max(...qnotes.map(n => n.gridEnd));
  // Round up to full bars
  const totalBars = Math.ceil(totalGrid / BAR_GRID);
  const paddedGrid = totalBars * BAR_GRID;

  // 2. Build flat event streams per staff
  const trebleEvents = buildStaffEvents(qnotes, 1, paddedGrid);
  const bassEvents   = buildStaffEvents(qnotes, 2, paddedGrid);

  // 3. Split events into bars
  function eventsForBar(events: XmlNote[], bar: number): XmlNote[] {
    const barStart = bar * BAR_GRID;
    const barEnd   = barStart + BAR_GRID;
    return events.filter(e => e.gridPos >= barStart && e.gridPos < barEnd);
  }

  // 4. Serialise
  let measuresXml = "";

  for (let bar = 0; bar < totalBars; bar++) {
    const attrXml = bar === 0 ? `
      <attributes>
        <divisions>${DIVISIONS}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>${BEATS_PER_BAR}</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${BPM}</per-minute></metronome></direction-type>
        <sound tempo="${BPM}"/>
      </direction>` : "";

    // Treble staff notes
    let trebleXml = "";
    for (const ev of eventsForBar(trebleEvents, bar)) {
      const ni = (ev as unknown as Record<string, number>)["noteIdx"] ?? 0;
      trebleXml += eventToXml(ev, ni);
    }
    if (!trebleXml) trebleXml = `\n        <note><rest/><duration>${BAR_GRID}</duration><type>whole</type><staff>1</staff></note>`;

    // Backup + bass staff notes
    const backupXml = `\n        <backup><duration>${BAR_GRID}</duration></backup>`;
    let bassXml = "";
    for (const ev of eventsForBar(bassEvents, bar)) {
      const ni = (ev as unknown as Record<string, number>)["noteIdx"] ?? 0;
      bassXml += eventToXml(ev, ni);
    }
    if (!bassXml) bassXml = `\n        <note><rest/><duration>${BAR_GRID}</duration><type>whole</type><staff>2</staff></note>`;

    measuresXml += `\n    <measure number="${bar + 1}">${attrXml}${trebleXml}${backupXml}${bassXml}\n    </measure>`;
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    ${measuresXml}
  </part>
</score-partwise>`;
}
