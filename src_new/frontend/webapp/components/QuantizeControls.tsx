"use client";

import type { QuantizeParams } from "@/lib/quantize";
import Slider from "@/components/Slider";

type Props = {
  params:      QuantizeParams;
  detectedBPM: number;
  onChange:    (p: QuantizeParams) => void;
};

const SUBDIVISIONS = [
  { value: 1, label: "1/4"  },
  { value: 2, label: "1/8"  },
  { value: 4, label: "1/16" },
  { value: 8, label: "1/32" },
];

export default function QuantizeControls({ params, detectedBPM, onChange }: Props) {
  function set<K extends keyof QuantizeParams>(key: K, value: QuantizeParams[K]) {
    onChange({ ...params, [key]: value });
  }

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 space-y-4">
      {/* Header + enable toggle */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-muted">
          Quantization
        </h2>
        <button
          onClick={() => set("enabled", !params.enabled)}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors
            ${params.enabled ? "bg-accent" : "bg-border"}`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform
              ${params.enabled ? "translate-x-4" : "translate-x-1"}`}
          />
        </button>
      </div>

      <div className={`space-y-4 transition-opacity ${params.enabled ? "opacity-100" : "opacity-40 pointer-events-none"}`}>

        {/* BPM */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-300">Tempo</span>
            <div className="flex items-center gap-3">
              <span className="text-[10px] text-muted font-mono">detected {detectedBPM} BPM</span>
              <span className="text-accent font-mono font-semibold">{params.bpm} BPM</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Slider
              min={40} max={240} step={0.5}
              value={params.bpm}
              onChange={v => set("bpm", v)}
              className="flex-1"
            />
            <button
              onClick={() => set("bpm", detectedBPM)}
              className="text-[10px] px-2 py-0.5 rounded-md bg-surface2 border border-border text-muted hover:text-slate-300 transition-colors whitespace-nowrap"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Grid subdivision */}
        <div className="space-y-1.5">
          <p className="text-xs text-slate-300">Grid</p>
          <div className="flex gap-2">
            {SUBDIVISIONS.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => set("subdivision", value)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors
                  ${params.subdivision === value
                    ? "bg-accent text-white"
                    : "bg-border text-muted hover:text-slate-200"}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Meter */}
        <div className="space-y-1.5">
          <p className="text-xs text-slate-300">Meter</p>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              value={params.timeSigNum}
              onChange={e => set("timeSigNum", Math.max(1, parseInt(e.target.value) || 1))}
              className="w-14 text-center bg-border text-slate-200 text-sm font-mono rounded-lg px-2 py-1 border border-border focus:outline-none focus:ring-1 focus:ring-accent"
            />
            <span className="text-muted text-base leading-none select-none">/</span>
            <input
              type="number"
              min={1}
              value={params.timeSigDen}
              onChange={e => set("timeSigDen", Math.max(1, parseInt(e.target.value) || 1))}
              className="w-14 text-center bg-border text-slate-200 text-sm font-mono rounded-lg px-2 py-1 border border-border focus:outline-none focus:ring-1 focus:ring-accent"
            />
          </div>
        </div>

        {/* Beat offset */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-300">Beat offset</span>
            <div className="flex items-center gap-3">
              <span className="text-accent font-mono">
                {params.beatOffset >= 0 ? "+" : ""}{(params.beatOffset * 1000).toFixed(0)} ms
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Slider
              min={-(60 / params.bpm)}
              max={60 / params.bpm}
              step={0.001}
              value={params.beatOffset}
              onChange={v => set("beatOffset", v)}
              className="flex-1"
            />
            <button
              onClick={() => set("beatOffset", 0)}
              className="text-[10px] px-2 py-0.5 rounded-md bg-surface2 border border-border text-muted hover:text-slate-300 transition-colors whitespace-nowrap"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Strength */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-300">Strength</span>
            <span className="text-accent font-mono">{Math.round(params.strength * 100)}%</span>
          </div>
          <Slider
            min={0} max={1} step={0.01}
            value={params.strength}
            onChange={v => set("strength", v)}
          />
        </div>

      </div>

      {params.enabled && (
        <p className="text-[10px] text-muted leading-relaxed">
          Beat grid shown on piano roll · score uses quantized positions.
        </p>
      )}
    </div>
  );
}
