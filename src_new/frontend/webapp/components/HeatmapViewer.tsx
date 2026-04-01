"use client";

import { useEffect, useRef } from "react";

// Must match PianoRollViewer constants exactly
const PITCH_MIN = 21;   // A0
const PITCH_MAX = 108;  // C8
const N_KEYS    = PITCH_MAX - PITCH_MIN + 1;  // 88
const ROW_H     = 6;    // px per semitone
const PX_PER_S  = 80;   // px per second

export type HeatmapMode = "frame" | "onset" | "both";

type Props = {
  pianoRoll: number[][];  // [T][88] frame probabilities
  onsetRoll: number[][];  // [T][88] onset probabilities
  fps:       number;
  duration:  number;
  mode:      HeatmapMode;
};

export default function HeatmapViewer({ pianoRoll, onsetRoll, fps, duration, mode }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const totalWidth  = Math.ceil(duration * PX_PER_S);
  const totalHeight = N_KEYS * ROW_H;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width  = totalWidth;
    canvas.height = totalHeight;

    const T = pianoRoll.length;
    if (T === 0) return;

    const imgData = ctx.createImageData(totalWidth, totalHeight);
    const data    = imgData.data;

    // Fractional pixels per frame (fps may be > PX_PER_S, so <1 px/frame)
    const pxPerFrame = totalWidth / T;

    for (let t = 0; t < T; t++) {
      const xStart = Math.floor(t       * pxPerFrame);
      const xEnd   = Math.floor((t + 1) * pxPerFrame);
      if (xStart >= totalWidth) break;

      for (let k = 0; k < N_KEYS; k++) {
        // k=0 → A0 (lowest pitch) → bottom row
        const row   = N_KEYS - 1 - k;
        const yStart = row * ROW_H;

        const fp = mode === "onset" ? 0 : pianoRoll[t][k];
        const op = mode === "frame" ? 0 : onsetRoll[t][k];

        // Perceptual dual-channel mapping:
        //   frame (sustained): cyan-blue  — (15, 30+fp*185, 25+fp*215)
        //   onset (attack):    warm amber — (15+op*210, 30+op*155, 25)
        //   both high → near-white
        const r = Math.min(255, 15  + op * 210);
        const g = Math.min(255, 30  + fp * 185 + op * 130);
        const b = Math.min(255, 25  + fp * 215);
        // Use full alpha — background comes from the canvas fill color
        const a = 255;

        for (let y = yStart; y < yStart + ROW_H; y++) {
          for (let x = xStart; x < Math.min(xEnd, totalWidth); x++) {
            const i = (y * totalWidth + x) * 4;
            data[i]     = r;
            data[i + 1] = g;
            data[i + 2] = b;
            data[i + 3] = a;
          }
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [pianoRoll, onsetRoll, fps, duration, mode, totalWidth, totalHeight]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display:         "block",
        width:           totalWidth,
        height:          totalHeight,
        imageRendering:  "pixelated",
      }}
    />
  );
}
