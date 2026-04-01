"use client";

import { useState, useRef, useMemo, useEffect } from "react";
import AudioDropzone from "@/components/AudioDropzone";
import TimeRangeSelector from "@/components/TimeRangeSelector";
import PianoRollViewer from "@/components/PianoRollViewer";
import DecodeControls from "@/components/DecodeControls";
import QuantizeControls from "@/components/QuantizeControls";
import { decodeNotes, DEFAULT_DECODE_PARAMS, type DecodeParams } from "@/lib/decode";
import { quantizeNotes, quantizeNotesFromBarTimes, estimateBPM, buildBeatGrid, buildGridFromBarTimes, DEFAULT_QUANTIZE_PARAMS, type QuantizeParams, type BeatGrid } from "@/lib/quantize";
import { Music2, Loader2, Download } from "lucide-react";
import type { HeatmapMode } from "@/components/HeatmapViewer";

const KEY_OPTIONS = [
  { label: "C major",  vex: "C"   }, { label: "G major",  vex: "G"   },
  { label: "D major",  vex: "D"   }, { label: "A major",  vex: "A"   },
  { label: "E major",  vex: "E"   }, { label: "B major",  vex: "B"   },
  { label: "F♯ major", vex: "F#"  }, { label: "C♯ major", vex: "C#"  },
  { label: "F major",  vex: "F"   }, { label: "B♭ major", vex: "Bb"  },
  { label: "E♭ major", vex: "Eb"  }, { label: "A♭ major", vex: "Ab"  },
  { label: "D♭ major", vex: "Db"  }, { label: "G♭ major", vex: "Gb"  },
  { label: "C♭ major", vex: "Cb"  },
  { label: "A minor",  vex: "Am"  }, { label: "E minor",  vex: "Em"  },
  { label: "B minor",  vex: "Bm"  }, { label: "F♯ minor", vex: "F#m" },
  { label: "C♯ minor", vex: "C#m" }, { label: "G♯ minor", vex: "G#m" },
  { label: "D minor",  vex: "Dm"  }, { label: "G minor",  vex: "Gm"  },
  { label: "C minor",  vex: "Cm"  }, { label: "F minor",  vex: "Fm"  },
  { label: "B♭ minor", vex: "Bbm" }, { label: "E♭ minor", vex: "Ebm" },
  { label: "A♭ minor", vex: "Abm" },
] as const;

export type Note = {
  pitch: number;
  start: number;
  end: number;
  midi_note: number;
  note_name: string;
};

export type AnalysisResult = {
  duration: number;
  fps: number;
  n_frames: number;
  piano_roll: number[][];
  onset_roll: number[][];
  notes: Note[];
};

type MidiExportFn = () => void;

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [startTime, setStartTime] = useState<number>(0);
  const [endTime, setEndTime] = useState<number>(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decodeParams, setDecodeParams] = useState<DecodeParams>(DEFAULT_DECODE_PARAMS);
  const [quantizeParams, setQuantizeParams] = useState<QuantizeParams>(DEFAULT_QUANTIZE_PARAMS);
  const [barTimes, setBarTimes] = useState<number[]>([]);
  const [keySig, setKeySig]     = useState("C");
  const [showHeatmap, setShowHeatmap]   = useState(false);
  const [heatmapMode, setHeatmapMode]   = useState<HeatmapMode>("both");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const midiExportRef = useRef<MidiExportFn | null>(null);

  function handleFileAccepted(f: File, duration: number) {
    setFile(f);
    setAudioDuration(duration);
    setStartTime(0);
    setEndTime(duration);
    setResult(null);
    setError(null);
  }

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const body = new FormData();
      body.append("file", file);
      body.append("start_time", String(startTime));
      body.append("end_time", String(endTime));

      const res = await fetch("/api/analyze", { method: "POST", body });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `Server error ${res.status}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // Re-decode client-side on every param change — no server round-trip
  const decodedNotes = useMemo(() => {
    if (!result) return [];
    return decodeNotes(result.onset_roll, result.piano_roll, result.fps, decodeParams);
  }, [result, decodeParams]);

  // Estimate BPM from onsets via autocorrelation
  const detectedBPM = useMemo(() => {
    if (!result || decodedNotes.length === 0) return 120;
    return estimateBPM(decodedNotes, result.fps, result.duration);
  }, [result, decodedNotes]);

  // Auto-populate BPM into quantize params whenever a new result arrives
  useEffect(() => {
    if (!result) return;
    setQuantizeParams(p => ({ ...p, bpm: detectedBPM }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  // Quantize note timings against the beat grid
  const quantizedNotes = useMemo(() => {
    if (!quantizeParams.enabled) return decodedNotes;
    if (barTimes.length >= 2) {
      return quantizeNotesFromBarTimes(
        decodedNotes, barTimes,
        quantizeParams.timeSigNum, quantizeParams.subdivision, quantizeParams.strength,
      );
    }
    return quantizeNotes(decodedNotes, quantizeParams);
  }, [decodedNotes, quantizeParams, barTimes]);

  // Beat grid for piano roll overlay — derives from barTimes so dragged barlines
  // immediately scale the beats within each bar.
  const beatGrid = useMemo((): BeatGrid | undefined => {
    if (!result || !quantizeParams.enabled) return undefined;
    if (barTimes.length >= 2) {
      return buildGridFromBarTimes(barTimes, quantizeParams.timeSigNum, quantizeParams.subdivision, result.duration);
    }
    return buildBeatGrid(quantizeParams.bpm, result.duration, quantizeParams.subdivision, quantizeParams.beatOffset);
  }, [result, quantizeParams, barTimes]);

  // Auto-compute bar positions from BPM + time signature.
  // barTimes[0] = beatOffset (bar 1 beat 1); subsequent bars spaced by barDur.
  // Resets whenever BPM, offset, or time sig changes; survives note-param changes.
  const autoBarTimes = useMemo(() => {
    if (!result) return [];
    const { bpm, beatOffset, timeSigNum, timeSigDen } = quantizeParams;
    const barDur = timeSigNum * (60 / bpm) * (4 / timeSigDen);
    if (barDur <= 0) return [];
    const times: number[] = [];
    // first bar index such that beatOffset + n*barDur is the earliest visible bar
    const firstN = Math.floor(-beatOffset / barDur);
    for (let n = firstN; ; n++) {
      const t = parseFloat((beatOffset + n * barDur).toFixed(6));
      if (t > result.duration + barDur * 0.01) break;
      if (t >= -barDur * 0.5) times.push(t);
    }
    return times;
  }, [result, quantizeParams.bpm, quantizeParams.beatOffset, quantizeParams.timeSigNum, quantizeParams.timeSigDen]);

  // Reset user-dragged bar times whenever the auto grid changes
  useEffect(() => {
    setBarTimes(autoBarTimes);
  }, [autoBarTimes]);

  const displayResult = result
    ? { ...result, notes: quantizedNotes }
    : null;

  return (
    <div className="max-w-6xl mx-auto px-6 py-10 space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-6xl font-extralight tracking-tight">Automatic Piano Music Transcription</h1>
          <p className="text-muted text-sm">Upload a piano recording and get an AI-generated transcription.</p>
        </div>
      </div>

      {/* Step 1 – Drop zone */}
      <section className="bg-surface border border-border rounded-2xl p-6 space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-muted">1 · Audio File</h2>
        <AudioDropzone onFileAccepted={handleFileAccepted} />
        {file && (
          <p className="text-xs text-muted">
            <span className="text-slate-300 font-medium">{file.name}</span>
            {" · "}{audioDuration.toFixed(1)} s
          </p>
        )}
      </section>

      {/* Step 2 – Time range */}
      {file && (
        <section className="bg-surface border border-border rounded-2xl p-6 space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-widest text-muted">2 · Time Range</h2>
          <TimeRangeSelector
            duration={audioDuration}
            start={startTime}
            end={endTime}
            onChange={(s, e) => { setStartTime(s); setEndTime(e); }}
          />
          <div className="flex justify-end">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="px-5 py-2 rounded-lg bg-accent hover:bg-accent-light disabled:opacity-50 disabled:cursor-not-allowed
                         font-semibold text-sm text-white transition-colors flex items-center gap-2"
            >
              {loading ? (
                <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Analysing…</>
              ) : (
                "Analyse"
              )}
            </button>
          </div>
        </section>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Results — controls + viewer in one section so they stay visible together */}
      {displayResult && (
        <section className="bg-surface border border-border rounded-2xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-widest text-muted">3 · Result</h2>
            <button
              onClick={() => midiExportRef.current?.()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent hover:bg-accent-light text-white transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Export MIDI
            </button>
          </div>

          {/* Decode + Quantize controls — above viewer so they stay in frame */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <DecodeControls params={decodeParams} onChange={setDecodeParams} />
            <QuantizeControls
              params={quantizeParams}
              detectedBPM={detectedBPM}
              onChange={setQuantizeParams}
            />
          </div>

          {/* Key selector */}
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold uppercase tracking-widest text-muted select-none">Key</span>
            <select
              value={keySig}
              onChange={e => setKeySig(e.target.value)}
              className="bg-surface border border-border rounded-lg px-2 py-1 text-sm text-white"
            >
              {KEY_OPTIONS.map(k => (
                <option key={k.vex} value={k.vex}>{k.label}</option>
              ))}
            </select>

            {/* Heatmap toggle */}
            <span className="text-xs font-semibold uppercase tracking-widest text-muted select-none ml-4">Heatmap</span>
            <button
              onClick={() => setShowHeatmap(v => !v)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors
                ${showHeatmap ? "bg-accent" : "bg-border"}`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform
                  ${showHeatmap ? "translate-x-4" : "translate-x-1"}`}
              />
            </button>

            {showHeatmap && (
              <select
                value={heatmapMode}
                onChange={e => setHeatmapMode(e.target.value as HeatmapMode)}
                className="bg-surface border border-border rounded-lg px-2 py-1 text-sm text-white"
              >
                <option value="both">Frame + Onset</option>
                <option value="frame">Frame only</option>
                <option value="onset">Onset only</option>
              </select>
            )}
          </div>

          <p className="text-xs text-muted">
            {displayResult.notes.length} notes · {displayResult.duration.toFixed(2)} s · {displayResult.fps.toFixed(2)} fps
          </p>

          <PianoRollViewer
            result={displayResult}
            beatGrid={beatGrid}
            barTimes={barTimes}
            onBarTimesChange={setBarTimes}
            timeSigNum={quantizeParams.timeSigNum}
            timeSigDen={quantizeParams.timeSigDen}
            bpm={quantizeParams.bpm}
            keySig={keySig}
            onRegisterMidiExport={fn => { midiExportRef.current = fn; }}
            showHeatmap={showHeatmap}
            heatmapMode={heatmapMode}
          />
        </section>
      )}

      {/* hidden audio element for duration probing */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
