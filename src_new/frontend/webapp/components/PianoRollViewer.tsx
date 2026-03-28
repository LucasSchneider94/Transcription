"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "@/app/page";

type Props = { result: AnalysisResult };

const PITCH_MIN = 21;   // A0
const PITCH_MAX = 108;  // C8
const N_KEYS    = PITCH_MAX - PITCH_MIN + 1;
const ROW_H     = 6;    // px per semitone
const PX_PER_S  = 80;   // px per second

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]); // semitone % 12

function isBlack(pitch: number) {
  return BLACK_KEYS.has(pitch % 12);
}

export default function PianoRollViewer({ result }: Props) {
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

    // background
    ctx.fillStyle = "#0f0f13";
    ctx.fillRect(0, 0, totalWidth, totalHeight);

    // horizontal key stripes
    for (let i = 0; i < N_KEYS; i++) {
      const pitch = PITCH_MAX - i;
      const y = i * ROW_H;
      ctx.fillStyle = isBlack(pitch) ? "#14141c" : "#1a1a24";
      ctx.fillRect(0, y, totalWidth, ROW_H);
    }

    // second grid lines
    ctx.strokeStyle = "#2a2a3a";
    ctx.lineWidth = 0.5;
    for (let s = 0; s <= result.duration; s++) {
      const x = s * PX_PER_S;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, totalHeight);
      ctx.stroke();
    }

    // notes
    for (const note of result.notes) {
      if (note.pitch < PITCH_MIN || note.pitch > PITCH_MAX) continue;
      const row   = PITCH_MAX - note.pitch;
      const x     = note.start * PX_PER_S;
      const w     = Math.max(2, (note.end - note.start) * PX_PER_S - 1);
      const y     = row * ROW_H + 1;
      const h     = ROW_H - 2;

      // rounded note rect
      ctx.fillStyle = isBlack(note.pitch) ? "#a89df9" : "#7c6af7";
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 2);
      ctx.fill();
    }
  }, [result, totalWidth, totalHeight]);

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
                  backgroundColor: isBlack(pitch) ? "#111" : "#1e1e2e",
                  borderRight: "2px solid #2a2a3a",
                  color: "#6b7280",
                }}
              >
                {label}
              </div>
            );
          })}
        </div>

        {/* scrollable canvas */}
        <div
          ref={containerRef}
          className="overflow-x-auto flex-1 rounded-lg"
          onScroll={(e) => setScrollX((e.target as HTMLDivElement).scrollLeft)}
        >
          <canvas
            ref={canvasRef}
            style={{ display: "block", imageRendering: "pixelated" }}
          />
        </div>
      </div>

      <p className="text-xs text-muted text-right">
        Scroll horizontally to navigate · {result.notes.length} notes
      </p>
    </div>
  );
}
