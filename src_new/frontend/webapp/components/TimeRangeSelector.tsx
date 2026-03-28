"use client";

type Props = {
  duration: number;
  start: number;
  end: number;
  onChange: (start: number, end: number) => void;
};

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

export default function TimeRangeSelector({ duration, start, end, onChange }: Props) {
  return (
    <div className="space-y-4">
      {/* dual slider track */}
      <div className="relative h-6 flex items-center">
        {/* filled range bar */}
        <div
          className="absolute h-1.5 bg-accent rounded-full pointer-events-none"
          style={{
            left: `${(start / duration) * 100}%`,
            right: `${100 - (end / duration) * 100}%`,
          }}
        />
        {/* base track */}
        <div className="absolute inset-x-0 h-1.5 bg-border rounded-full -z-10" />

        {/* start thumb */}
        <input
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={start}
          onChange={(e) => {
            const v = Math.min(Number(e.target.value), end - 0.5);
            onChange(v, end);
          }}
          className="absolute inset-0 w-full opacity-0 cursor-pointer h-6"
          style={{ zIndex: 2 }}
        />
        {/* end thumb */}
        <input
          type="range"
          min={0}
          max={duration}
          step={0.1}
          value={end}
          onChange={(e) => {
            const v = Math.max(Number(e.target.value), start + 0.5);
            onChange(start, v);
          }}
          className="absolute inset-0 w-full opacity-0 cursor-pointer h-6"
          style={{ zIndex: 3 }}
        />

        {/* visible thumbs */}
        <div
          className="absolute w-4 h-4 bg-accent rounded-full border-2 border-white shadow -translate-x-1/2 pointer-events-none"
          style={{ left: `${(start / duration) * 100}%`, zIndex: 4 }}
        />
        <div
          className="absolute w-4 h-4 bg-accent rounded-full border-2 border-white shadow -translate-x-1/2 pointer-events-none"
          style={{ left: `${(end / duration) * 100}%`, zIndex: 4 }}
        />
      </div>

      {/* numeric inputs */}
      <div className="flex items-center gap-4 text-sm">
        <label className="flex items-center gap-2 text-muted">
          Start
          <input
            type="number"
            min={0}
            max={end - 0.5}
            step={0.1}
            value={start.toFixed(1)}
            onChange={(e) => onChange(Math.max(0, Number(e.target.value)), end)}
            className="w-20 bg-background border border-border rounded-lg px-2 py-1 text-slate-200 text-xs"
          />
          <span className="text-xs">{fmt(start)}</span>
        </label>

        <span className="text-border">–</span>

        <label className="flex items-center gap-2 text-muted">
          End
          <input
            type="number"
            min={start + 0.5}
            max={duration}
            step={0.1}
            value={end.toFixed(1)}
            onChange={(e) => onChange(start, Math.min(duration, Number(e.target.value)))}
            className="w-20 bg-background border border-border rounded-lg px-2 py-1 text-slate-200 text-xs"
          />
          <span className="text-xs">{fmt(end)}</span>
        </label>

        <span className="ml-auto text-xs text-muted">
          Duration: <span className="text-slate-300">{(end - start).toFixed(1)} s</span>
        </span>
      </div>
    </div>
  );
}
