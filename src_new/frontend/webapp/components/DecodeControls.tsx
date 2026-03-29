"use client";

import type { DecodeParams } from "@/lib/decode";

type Props = {
  params: DecodeParams;
  onChange: (p: DecodeParams) => void;
};

type SliderDef = {
  key: keyof DecodeParams;
  label: string;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
};

const SLIDERS: SliderDef[] = [
  { key: "onset_threshold", label: "Onset threshold",   min: 0.1,  max: 0.95, step: 0.01 },
  { key: "frame_threshold", label: "Frame threshold",   min: 0.1,  max: 0.95, step: 0.01 },
  { key: "refractory_ms",   label: "Re-strike gap",     min: 10,   max: 300,  step: 5, format: (v) => `${v} ms` },
  { key: "min_note_ms",     label: "Min note length",   min: 10,   max: 200,  step: 5, format: (v) => `${v} ms` },
  { key: "frame_smoothing", label: "Frame smoothing",   min: 0,    max: 5,    step: 1, format: (v) => v === 0 ? "off" : `±${v} frames` },
];

export default function DecodeControls({ params, onChange }: Props) {
  function set<K extends keyof DecodeParams>(key: K, value: DecodeParams[K]) {
    onChange({ ...params, [key]: value });
  }

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 space-y-4">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-muted">
        Decoding parameters
      </h2>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {SLIDERS.map(({ key, label, min, max, step, format }) => {
          const val = params[key] as number;
          return (
            <div key={key} className="space-y-1">
              <div className="flex justify-between text-xs text-slate-300">
                <span>{label}</span>
                <span className="text-accent font-mono">
                  {format ? format(val) : val.toFixed(step < 0.1 ? 2 : 0)}
                </span>
              </div>
              <input
                type="range"
                min={min} max={max} step={step}
                value={val}
                onChange={(e) => set(key, parseFloat(e.target.value) as DecodeParams[typeof key])}
                className="w-full accent-[color:var(--color-accent)] h-1.5 rounded-full cursor-pointer"
              />
            </div>
          );
        })}

        {/* Onset gating toggle */}
        <div className="flex items-center justify-between sm:col-span-2">
          <span className="text-xs text-slate-300">Onset gating</span>
          <button
            onClick={() => set("onset_gating", !params.onset_gating)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors
              ${params.onset_gating ? "bg-accent" : "bg-border"}`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform
                ${params.onset_gating ? "translate-x-4" : "translate-x-1"}`}
            />
          </button>
        </div>
      </div>

      <p className="text-[10px] text-muted leading-relaxed">
        Changes are applied instantly — no re-inference needed.
      </p>
    </div>
  );
}
