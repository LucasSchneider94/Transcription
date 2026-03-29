"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "@/app/page";
import type { BeatGrid } from "@/lib/quantize";

type Props = {
  result:             AnalysisResult;
  beatGrid?:          BeatGrid;
  barTimes?:          number[];
  onBarTimesChange?:  (times: number[]) => void;
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

export default function PianoRollViewer({ result, beatGrid, barTimes, onBarTimesChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollX, setScrollX] = useState(0);

  const totalWidth  = Math.ceil(result.duration * PX_PER_S);
  const totalHeight = N_KEYS * ROW_H;

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

    // Truncate each note at the next onset on the same pitch so re-strikes
    // are visible as a gap rather than a seamless continuation.
    const GAP_S = 4 / PX_PER_S; // 4 px gap regardless of zoom
    const byPitch = new Map<number, typeof result.notes>();
    for (const note of result.notes) {
      if (!byPitch.has(note.pitch)) byPitch.set(note.pitch, []);
      byPitch.get(note.pitch)!.push(note);
    }
    const truncated: { pitch: number; start: number; end: number }[] = [];
    for (const [, group] of byPitch) {
      const sorted = [...group].sort((a, b) => a.start - b.start);
      for (let i = 0; i < sorted.length; i++) {
        const next = sorted[i + 1];
        const end = next
          ? Math.min(sorted[i].end, next.start - GAP_S)
          : sorted[i].end;
        truncated.push({ pitch: sorted[i].pitch, start: sorted[i].start, end });
      }
    }

    // notes
    for (const note of truncated) {
      if (note.pitch < PITCH_MIN || note.pitch > PITCH_MAX) continue;
      const row = PITCH_MAX - note.pitch;
      const x   = note.start * PX_PER_S;
      const w   = Math.max(2, (note.end - note.start) * PX_PER_S - 1);
      const y   = row * ROW_H + 1;
      const h   = ROW_H - 2;

      ctx.fillStyle = isBlack(note.pitch) ? clrNoteBlack : clrNoteWhite;
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 2);
      ctx.fill();
    }
  }, [result, totalWidth, totalHeight, beatGrid]);

  return (
    <div className="space-y-2">
      {/* keyboard labels on left, scroll in sync */}
      <div className="flex">
        {/* piano keyboard strip */}
        <div className="flex-shrink-0 w-10 relative" style={{ height: totalHeight }}>
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

        {/* scrollable canvas + draggable barlines */}
        <div
          ref={containerRef}
          className="overflow-x-auto flex-1 rounded-lg"
          onScroll={(e) => setScrollX((e.target as HTMLDivElement).scrollLeft)}
        >
          <div className="relative" style={{ width: totalWidth, height: totalHeight }}>
            <canvas
              ref={canvasRef}
              style={{ display: "block", position: "absolute", top: 0, left: 0, imageRendering: "pixelated" }}
            />
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

      <p className="text-xs text-muted text-right">
        Scroll horizontally to navigate · {result.notes.length} notes
      </p>
    </div>
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
          // amber for bar 1, green for others
          backgroundColor: isFirst ? "rgba(250,176,5,0.9)" : "rgba(61,170,114,0.75)",
          pointerEvents:   "none",
        }}
      />
    </div>
  );
}
