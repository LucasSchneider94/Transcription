"use client";

import { useState, useRef, useMemo } from "react";
import AudioDropzone from "@/components/AudioDropzone";
import TimeRangeSelector from "@/components/TimeRangeSelector";
import PianoRollViewer from "@/components/PianoRollViewer";
import ProportionalScoreViewer from "@/components/ProportionalScoreViewer";
import DecodeControls from "@/components/DecodeControls";
import { decodeNotes, DEFAULT_DECODE_PARAMS, type DecodeParams } from "@/lib/decode";
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

  const displayResult = result
    ? { ...result, notes: decodedNotes }
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

      {/* Results */}
      {displayResult && (
        <>
          {/* Decoding controls */}
          <DecodeControls params={decodeParams} onChange={setDecodeParams} />

          <section className="bg-surface border border-border rounded-2xl p-6 space-y-4">
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

            <p className="text-xs text-muted">
              {displayResult.notes.length} notes · {displayResult.duration.toFixed(2)} s · {displayResult.fps.toFixed(2)} fps
            </p>

            {viewMode === "piano_roll" ? (
              <PianoRollViewer result={displayResult} />
            ) : (
              <ProportionalScoreViewer notes={displayResult.notes} duration={displayResult.duration} />
            )}
          </section>
        </>
      )}

      {/* hidden audio element for duration probing */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
