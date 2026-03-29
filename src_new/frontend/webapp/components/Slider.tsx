"use client";

import { useNoScrollOnFocus } from "@/lib/useNoScrollOnFocus";

type Props = {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  className?: string;
};

export default function Slider({ min, max, step, value, onChange, className = "w-full" }: Props) {
  const noScroll = useNoScrollOnFocus();
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={`relative h-6 flex items-center ${className}`}>
      {/* base track */}
      <div className="absolute inset-x-0 h-1.5 rounded-full -z-10" style={{ background: 'var(--slider-bg)' }} />
      {/* filled portion */}
      <div
        className="absolute h-1.5 bg-accent rounded-full pointer-events-none"
        style={{ left: 0, width: `${pct}%` }}
      />
      {/* invisible input for interaction */}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        {...noScroll}
        className="absolute inset-0 w-full opacity-0 cursor-pointer h-6"
      />
      {/* visible thumb */}
      <div
        className="absolute w-4 h-4 bg-accent rounded-full border-2 border-white shadow -translate-x-1/2 pointer-events-none"
        style={{ left: `${pct}%` }}
      />
    </div>
  );
}
