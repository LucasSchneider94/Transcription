"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisResult, Note } from "@/app/page";
import type { BeatGrid } from "@/lib/quantize";
import LinearScoreViewer, { LINEAR_SCORE_H, LINE_SPACING, TREBLE_Y as CLEF_TREBLE_Y, BASS_Y as CLEF_BASS_Y } from "@/components/LinearScoreViewer";

type Props = {
  result:                AnalysisResult;
  beatGrid?:             BeatGrid;
  barTimes?:             number[];
  onBarTimesChange?:     (times: number[]) => void;
  timeSigNum?:           number;
  timeSigDen?:           number;
  bpm?:                  number;
  keySig?:               string;
  onRegisterMidiExport?: (fn: () => void) => void;
};

const PITCH_MIN = 21;   // A0
const PITCH_MAX = 108;  // C8
const N_KEYS    = PITCH_MAX - PITCH_MIN + 1;
const ROW_H     = 6;    // px per semitone
const PX_PER_S  = 80;   // px per second

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]); // semitone % 12

function isBlack(pitch: number) {
  return BLACK_KEYS.has(pitch % 12);
}

export default function PianoRollViewer({
  result, beatGrid, barTimes, onBarTimesChange,
  timeSigNum = 4, timeSigDen = 4, bpm = 120, keySig = "C", onRegisterMidiExport,
}: Props) {
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const containerRef  = useRef<HTMLDivElement>(null);
  const [scrollX, setScrollX]       = useState(0);
  const [editNotes, setEditNotes]   = useState<Note[]>(() => result.notes);
  const [selected, setSelected]     = useState<number | null>(null);

  // Sync editNotes when upstream notes change (new result / re-quantize)
  useEffect(() => {
    setEditNotes(result.notes);
    setSelected(null);
  }, [result.notes]);

  // Register MIDI export callback with parent — uses editNotes so edits are exported
  useEffect(() => {
    onRegisterMidiExport?.(() => exportMidi(editNotes, timeSigNum, timeSigDen, bpm));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editNotes, timeSigNum, timeSigDen, bpm]);

  const totalWidth  = Math.ceil(result.duration * PX_PER_S);
  const totalHeight = N_KEYS * ROW_H;

  // ─── edit helpers ───────────────────────────────────────────────────────────
  const updateNote = useCallback((idx: number, patch: Partial<Note>) => {
    setEditNotes(prev => {
      const next = [...prev];
      next[idx]  = { ...next[idx], ...patch };
      return next;
    });
  }, []);

  const deleteNote = useCallback((idx: number) => {
    setEditNotes(prev => prev.filter((_, i) => i !== idx));
    setSelected(null);
  }, []);

  const snapToGrid = useCallback((t: number): number => {
    const subs = beatGrid?.subs;
    if (!subs || subs.length < 2) return t;
    let best = t, minD = Infinity;
    for (const s of subs) {
      const d = Math.abs(s - t);
      if (d < minD) { minD = d; best = s; }
      if (s > t + 1) break;
    }
    return best;
  }, [beatGrid]);

  // Keyboard handler — attached to outer container
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (selected === null) return;
    const note   = editNotes[selected];
    const subs   = beatGrid?.subs;
    const step   = subs && subs.length >= 2 ? subs[1] - subs[0] : 0.0625;
    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        updateNote(selected, { start: Math.max(0, note.start - step), end: Math.max(step * 2, note.end - step) });
        break;
      case "ArrowRight":
        e.preventDefault();
        updateNote(selected, { start: note.start + step, end: note.end + step });
        break;
      case "ArrowUp":
        e.preventDefault();
        updateNote(selected, { end: note.end + step });
        break;
      case "ArrowDown":
        e.preventDefault();
        updateNote(selected, { end: Math.max(note.start + step, note.end - step) });
        break;
      case "Delete":
      case "Backspace":
        e.preventDefault();
        deleteNote(selected);
        break;
    }
  }, [selected, editNotes, beatGrid, updateNote, deleteNote]);

  // Pre-compute display ends (gap between consecutive same-pitch notes)
  const GAP_S = 4 / PX_PER_S;
  const displayEnds = useMemo(() => {
    const byPitch = new Map<number, { idx: number; start: number; end: number }[]>();
    editNotes.forEach((n, idx) => {
      if (!byPitch.has(n.pitch)) byPitch.set(n.pitch, []);
      byPitch.get(n.pitch)!.push({ idx, start: n.start, end: n.end });
    });
    const out = new Map<number, number>(); // idx → display end
    for (const [, group] of byPitch) {
      const sorted = [...group].sort((a, b) => a.start - b.start);
      for (let i = 0; i < sorted.length; i++) {
        const next = sorted[i + 1];
        out.set(sorted[i].idx, next
          ? Math.min(sorted[i].end, next.start - GAP_S)
          : sorted[i].end);
      }
    }
    return out;
  }, [editNotes, GAP_S]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width  = totalWidth;
    canvas.height = totalHeight;

    // Read palette from CSS custom properties so it stays in sync with globals.css
    const style = getComputedStyle(document.documentElement);
    const clrBg           = style.getPropertyValue("--roll-bg").trim()           || "#0d0d0d";
    const clrStripeBlack  = style.getPropertyValue("--roll-stripe-black").trim() || "#131313";
    const clrStripeWhite  = style.getPropertyValue("--roll-stripe-white").trim() || "#181818";
    const clrGrid         = style.getPropertyValue("--roll-grid").trim()         || "#1a281a";
    const clrBeat         = style.getPropertyValue("--roll-beat").trim()         || "rgba(61,170,114,0.20)";
    const clrBar          = style.getPropertyValue("--roll-bar").trim()          || "rgba(61,170,114,0.45)";
    const clrNoteWhite    = style.getPropertyValue("--roll-note-white").trim()   || "#3daa72";
    const clrNoteBlack    = style.getPropertyValue("--roll-note-black").trim()   || "#1d9080";

    // background
    ctx.fillStyle = clrBg;
    ctx.fillRect(0, 0, totalWidth, totalHeight);

    // horizontal key stripes
    for (let i = 0; i < N_KEYS; i++) {
      const pitch = PITCH_MAX - i;
      const y = i * ROW_H;
      ctx.fillStyle = isBlack(pitch) ? clrStripeBlack : clrStripeWhite;
      ctx.fillRect(0, y, totalWidth, ROW_H);
    }

    // vertical grid lines
    if (beatGrid) {
      // sub-grid (lightest) — only draw if spacing is at least 6px to avoid clutter
      const subSpacingPx = (beatGrid.subs[1] ?? 0 - (beatGrid.subs[0] ?? 0)) * PX_PER_S;
      if (subSpacingPx >= 6) {
        ctx.strokeStyle = clrGrid;
        ctx.lineWidth = 0.5;
        for (const t of beatGrid.subs) {
          const x = t * PX_PER_S;
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, totalHeight); ctx.stroke();
        }
      }
      // beat lines
      ctx.strokeStyle = clrBeat;
      ctx.lineWidth = 1;
      for (const t of beatGrid.beats) {
        const x = t * PX_PER_S;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, totalHeight); ctx.stroke();
      }
      // bar lines — drawn as draggable overlays, not on canvas
    } else {
      // fallback: 1-second lines
      ctx.strokeStyle = clrGrid;
      ctx.lineWidth = 0.5;
      for (let s = 0; s <= result.duration; s++) {
        const x = s * PX_PER_S;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, totalHeight); ctx.stroke();
      }
    }

    // Notes are rendered as DOM overlay divs — canvas only draws background/grid
  }, [result, totalWidth, totalHeight, beatGrid]);

  return (
    <div
      className="space-y-2"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
      style={{ outline: "none" }}
    >
      {/* keyboard labels on left, scroll in sync */}
      <div className="flex">
        {/* left strip: clef labels + piano keys */}
        <div className="flex-shrink-0" style={{ width: CLEF_STRIP_W }}>
          {/* clef placeholder — VexFlow render, must match LinearScoreViewer layout */}
          <ClefStrip keySig={keySig} />
          {/* piano keyboard strip */}
          <div className="relative" style={{ height: totalHeight }}>
            {Array.from({ length: N_KEYS }, (_, i) => {
              const pitch = PITCH_MAX - i;
              const label = pitch % 12 === 0 ? `C${Math.floor(pitch / 12) - 1}` : "";
              return (
                <div
                  key={pitch}
                  className="absolute right-0 flex items-center justify-end pr-1 text-[9px]"
                  style={{
                  top: i * ROW_H,
                  height: ROW_H,
                  width: "100%",
                  backgroundColor: isBlack(pitch) ? "var(--roll-stripe-black)" : "var(--roll-stripe-white)",
                  borderRight: "2px solid var(--color-border)",
                  color: "#6b7280",
                }}
              >
                {label}
              </div>
            );
          })}
          </div>
        </div>

        {/* scrollable canvas + (optional) aligned score above */}
        <div
          ref={containerRef}
          className="overflow-x-auto flex-1 rounded-lg"
          onScroll={(e) => setScrollX((e.target as HTMLDivElement).scrollLeft)}
        >
          <div style={{ width: totalWidth }}>
            {/* Linear score strip — always visible, shares scroll */}
            <LinearScoreViewer
              notes={editNotes}
              barTimes={barTimes ?? []}
              timeSigNum={timeSigNum}
              timeSigDen={timeSigDen}
              totalWidth={totalWidth}
              keySig={keySig}
            />

            {/* Piano roll canvas + note overlay + draggable barlines */}
            <div
              className="relative"
              style={{ width: totalWidth, height: totalHeight }}
              onClick={() => setSelected(null)}
            >
              <canvas
                ref={canvasRef}
                style={{ display: "block", position: "absolute", top: 0, left: 0, imageRendering: "pixelated" }}
              />
              {/* Note divs */}
              {editNotes.map((note, idx) => {
                if (note.pitch < PITCH_MIN || note.pitch > PITCH_MAX) return null;
                return (
                  <NoteBlock
                    key={idx}
                    note={note}
                    displayEnd={displayEnds.get(idx) ?? note.end}
                    isSelected={selected === idx}
                    onSelect={() => setSelected(idx)}
                    onMove={(newStart, newPitch) => {
                      const dur = note.end - note.start;
                      const snapped = snapToGrid(Math.max(0, newStart));
                      const p = Math.max(PITCH_MIN, Math.min(PITCH_MAX, newPitch));
                      updateNote(idx, { start: snapped, end: snapped + dur, pitch: p, midi_note: p });
                    }}
                    onResize={(newEnd) => {
                      const snapped = snapToGrid(newEnd);
                      updateNote(idx, { end: Math.max(note.start + 0.05, snapped) });
                    }}
                  />
                );
              })}
              {barTimes?.map((t, i) => (
                <BarHandle
                  key={i}
                  time={Math.max(0, Math.min(result.duration, t))}
                  height={totalHeight}
                  isFirst={i === 0}
                  onDrag={newTime => {
                    if (!onBarTimesChange) return;
                    const updated = [...barTimes];
                    updated[i] = Math.max(0, Math.min(result.duration, newTime));
                    onBarTimesChange(updated);
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      <p className="text-xs text-muted text-right">
        Scroll horizontally to navigate · {editNotes.length} notes
        {selected !== null && " · selected — ←/→ move, ↑/↓ lengthen/shorten, Del removes"}
      </p>
    </div>
  );
}

// ─── Fixed clef strip (VexFlow, left of scroll area) ──────────────────────────

// These must mirror the constants in LinearScoreViewer.tsx
// These mirror LinearScoreViewer's TREBLE_Y, BASS_Y — imported above
const CLEF_STRIP_W  = 120;

function ClefStrip({ keySig }: { keySig: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    (async () => {
      const { Renderer, Stave } = await import("vexflow");
      el.innerHTML = "";
      const renderer = new Renderer(el, Renderer.Backends.SVG);
      renderer.resize(CLEF_STRIP_W, LINEAR_SCORE_H);
      const ctx = renderer.getContext();
      const opts = { spacing_between_lines_px: LINE_SPACING };
      const ts = new Stave(0, CLEF_TREBLE_Y, CLEF_STRIP_W, opts);
      const bs = new Stave(0, CLEF_BASS_Y,   CLEF_STRIP_W, opts);
      ts.addClef("treble"); ts.addKeySignature(keySig);
      bs.addClef("bass");   bs.addKeySignature(keySig);
      ts.setContext(ctx).draw();
      bs.setContext(ctx).draw();
      const svg = el.querySelector("svg");
      if (svg) { svg.style.background = "white"; svg.style.overflow = "visible"; }
    })().catch(e => console.warn("ClefStrip:", e));
  }, [keySig]);

  return (
    <div
      ref={ref}
      style={{ width: CLEF_STRIP_W, height: LINEAR_SCORE_H, flexShrink: 0 }}
      className="bg-white border-r border-gray-300 [&_text]:fill-black [&_path]:stroke-black select-none"
    />
  );
}

// ─── Draggable barline handle ────────────────────────────────────────────────

function BarHandle({
  time, height, isFirst, onDrag,
}: {
  time: number; height: number; isFirst: boolean; onDrag: (t: number) => void;
}) {
  const startXRef    = useRef<number>(0);
  const startTimeRef = useRef<number>(time);

  function handleMouseDown(e: React.MouseEvent) {
    e.preventDefault();
    startXRef.current    = e.clientX;
    startTimeRef.current = time;

    function onMove(me: MouseEvent) {
      onDrag(startTimeRef.current + (me.clientX - startXRef.current) / PX_PER_S);
    }
    function onUp() {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
  }

  return (
    <div
      onMouseDown={handleMouseDown}
      title={`${isFirst ? "Bar 1 · " : ""}${time.toFixed(3)} s`}
      style={{
        position:       "absolute",
        left:           time * PX_PER_S - 5,
        top:            0,
        width:          10,
        height,
        cursor:         "ew-resize",
        zIndex:         10,
        display:        "flex",
        justifyContent: "center",
        userSelect:     "none",
      }}
    >
      <div
        style={{
          width:           isFirst ? 2 : 1.5,
          height:          "100%",
          backgroundColor: isFirst ? "var(--color-accent-light)" : "var(--color-accent-cool)",
          pointerEvents:   "none",
        }}
      />
    </div>
  );
}

// ─── Note block (DOM, editable) ──────────────────────────────────────────────

function NoteBlock({
  note, displayEnd, isSelected, onSelect, onMove, onResize,
}: {
  note:        Note;
  displayEnd:  number;
  isSelected:  boolean;
  onSelect:    () => void;
  onMove:      (newStart: number, newPitch: number) => void;
  onResize:    (newEnd: number) => void;
}) {
  const row = PITCH_MAX - note.pitch;
  const x   = note.start * PX_PER_S;
  const w   = Math.max(3, (displayEnd - note.start) * PX_PER_S - 1);
  const y   = row * ROW_H + 1;
  const h   = ROW_H - 2;

  const clr = isSelected
    ? "var(--note-highlight)"
    : isBlack(note.pitch) ? "var(--roll-note-black)" : "var(--roll-note-white)";

  function handleBodyMouseDown(e: React.MouseEvent) {
    e.stopPropagation();
    onSelect();
    const startX     = e.clientX;
    const startY     = e.clientY;
    const startTime  = note.start;
    const startPitch = note.pitch;

    function onMove_(me: MouseEvent) {
      const dt    = (me.clientX - startX) / PX_PER_S;
      const dp    = -Math.round((me.clientY - startY) / ROW_H);
      onMove(startTime + dt, startPitch + dp);
    }
    function onUp() {
      window.removeEventListener("mousemove", onMove_);
      window.removeEventListener("mouseup",   onUp);
    }
    window.addEventListener("mousemove", onMove_);
    window.addEventListener("mouseup",   onUp);
  }

  function handleResizeMouseDown(e: React.MouseEvent) {
    e.stopPropagation();
    onSelect();
    const startX   = e.clientX;
    const startEnd = note.end;

    function onMove_(me: MouseEvent) {
      onResize(startEnd + (me.clientX - startX) / PX_PER_S);
    }
    function onUp() {
      window.removeEventListener("mousemove", onMove_);
      window.removeEventListener("mouseup",   onUp);
    }
    window.addEventListener("mousemove", onMove_);
    window.addEventListener("mouseup",   onUp);
  }

  return (
    <div
      onMouseDown={handleBodyMouseDown}
      onClick={(e) => e.stopPropagation()}
      style={{
        position:        "absolute",
        left:            x,
        top:             y,
        width:           w,
        height:          h,
        backgroundColor: clr,
        borderRadius:    2,
        cursor:          "grab",
        boxSizing:       "border-box",
        border:          isSelected ? "1px solid var(--note-highlight)" : "none",
        zIndex:          5,
      }}
    >
      {/* resize handle — right 5px */}
      <div
        onMouseDown={handleResizeMouseDown}
        style={{
          position:  "absolute",
          right:     0,
          top:       0,
          width:     5,
          height:    "100%",
          cursor:    "ew-resize",
        }}
      />
    </div>
  );
}

// ─── MIDI export ─────────────────────────────────────────────────────────────
// Hand-rolled minimal Standard MIDI File (SMF format 0) writer.
// No external library — keeps the bundle lean and avoids SSR issues.

function writeMidi(notes: AnalysisResult["notes"], timeSigNum: number, timeSigDen: number, bpm: number): Uint8Array {
  const PPQ         = 480;   // pulses per quarter note
  const TEMPO       = Math.round(60_000_000 / bpm); // microseconds per quarter note
  const beatsPerSec = bpm / 60;

  function varLen(n: number): number[] {
    const bytes: number[] = [];
    bytes.unshift(n & 0x7f);
    n >>= 7;
    while (n > 0) { bytes.unshift((n & 0x7f) | 0x80); n >>= 7; }
    return bytes;
  }

  function writeU32(n: number): number[] {
    return [(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
  }

  function writeU16(n: number): number[] {
    return [(n >> 8) & 0xff, n & 0xff];
  }

  // Build events: [tick, ...bytes]
  type Ev = [number, ...number[]];
  const events: Ev[] = [];

  // Tempo
  events.push([0, 0xff, 0x51, 0x03, (TEMPO >> 16) & 0xff, (TEMPO >> 8) & 0xff, TEMPO & 0xff]);

  // Time signature
  const log2Den = Math.round(Math.log2(timeSigDen));
  events.push([0, 0xff, 0x58, 0x04, timeSigNum, log2Den, 24, 8]);

  for (const note of notes) {
    const startTick = Math.round(note.start * PPQ * beatsPerSec);
    const endTick   = Math.round(note.end   * PPQ * beatsPerSec);
    const vel = 80;
    events.push([startTick, 0x90, note.midi_note & 0x7f, vel]);
    events.push([endTick,   0x80, note.midi_note & 0x7f, 0]);
  }

  events.sort((a, b) => a[0] - b[0]);

  // Convert to delta-time
  const track: number[] = [];
  let lastTick = 0;
  for (const [tick, ...msg] of events) {
    track.push(...varLen(tick - lastTick), ...msg);
    lastTick = tick;
  }
  // End of track
  track.push(...varLen(0), 0xff, 0x2f, 0x00);

  const header = [
    0x4d, 0x54, 0x68, 0x64,  // MThd
    ...writeU32(6),            // chunk length
    ...writeU16(0),            // format 0
    ...writeU16(1),            // 1 track
    ...writeU16(PPQ),          // PPQ
    0x4d, 0x54, 0x72, 0x6b,  // MTrk
    ...writeU32(track.length),
    ...track,
  ];

  return new Uint8Array(header);
}

function exportMidi(notes: AnalysisResult["notes"], timeSigNum: number, timeSigDen: number, bpm: number) {
  const data = writeMidi(notes, timeSigNum, timeSigDen, bpm);
  const buf  = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
  const blob = new Blob([buf], { type: "audio/midi" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "transcription.mid";
  a.click();
  URL.revokeObjectURL(url);
}
