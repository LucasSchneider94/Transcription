"use client";

import { useState, useRef, useMemo, useEffect } from "react";
import AudioDropzone from "@/components/AudioDropzone";
import TimeRangeSelector from "@/components/TimeRangeSelector";
import PianoRollViewer from "@/components/PianoRollViewer";
import ProportionalScoreViewer from "@/components/ProportionalScoreViewer";
import DecodeControls from "@/components/DecodeControls";
import QuantizeControls from "@/components/QuantizeControls";
import { decodeNotes, DEFAULT_DECODE_PARAMS, type DecodeParams } from "@/lib/decode";
import { quantizeNotes, quantizeNotesFromBarTimes, estimateBPM, buildBeatGrid, buildGridFromBarTimes, DEFAULT_QUANTIZE_PARAMS, type QuantizeParams, type BeatGrid } from "@/lib/quantize";
import { Music2, Loader2 } from "lucide-react";

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

type ViewMode = "piano_roll" | "score";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [startTime, setStartTime] = useState<number>(0);
  const [endTime, setEndTime] = useState<number>(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("piano_roll");
  const [decodeParams, setDecodeParams] = useState<DecodeParams>(DEFAULT_DECODE_PARAMS);
  const [quantizeParams, setQuantizeParams] = useState<QuantizeParams>(DEFAULT_QUANTIZE_PARAMS);
  const [barTimes, setBarTimes] = useState<number[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

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
        <Music2 className="text-accent w-8 h-8" />
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Piano Transcription</h1>
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
        </section>
      )}

      {/* Step 3 – Analyse button */}
      {file && (
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="w-full py-3 rounded-xl bg-accent hover:bg-accent-light disabled:opacity-50 disabled:cursor-not-allowed
                     font-semibold text-white transition-colors flex items-center justify-center gap-2"
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Analysing…</>
          ) : (
            "Analyse"
          )}
        </button>
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
            <div className="flex gap-2">
              {(["piano_roll", "score"] as ViewMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setViewMode(m)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors
                    ${viewMode === m
                      ? "bg-accent text-white"
                      : "bg-border text-muted hover:text-slate-200"}`}
                >
                  {m === "piano_roll" ? "Piano Roll" : "Score"}
                </button>
              ))}
            </div>
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

          <p className="text-xs text-muted">
            {displayResult.notes.length} notes · {displayResult.duration.toFixed(2)} s · {displayResult.fps.toFixed(2)} fps
          </p>

          {viewMode === "piano_roll" ? (
            <PianoRollViewer
              result={displayResult}
              beatGrid={beatGrid}
              barTimes={barTimes}
              onBarTimesChange={setBarTimes}
            />
          ) : (
            <ProportionalScoreViewer
              notes={displayResult.notes}
              duration={displayResult.duration}
              bpm={quantizeParams.bpm}
              beatOffset={quantizeParams.beatOffset}
              timeSigNum={quantizeParams.timeSigNum}
              timeSigDen={quantizeParams.timeSigDen}
            />
          )}
        </section>
      )}

      {/* hidden audio element for duration probing */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
