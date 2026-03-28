"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, FileAudio } from "lucide-react";

type Props = {
  onFileAccepted: (file: File, duration: number) => void;
};

export default function AudioDropzone({ onFileAccepted }: Props) {
  const [isDragActive, setIsDragActive] = useState(false);

  const onDrop = useCallback(
    (accepted: File[]) => {
      const f = accepted[0];
      if (!f) return;
      // probe duration via a hidden Audio element
      const url = URL.createObjectURL(f);
      const audio = new Audio(url);
      audio.addEventListener("loadedmetadata", () => {
        onFileAccepted(f, audio.duration);
        URL.revokeObjectURL(url);
      });
    },
    [onFileAccepted]
  );

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: { "audio/*": [".wav", ".mp3", ".flac", ".ogg", ".m4a"] },
    maxFiles: 1,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
    onDropAccepted: () => setIsDragActive(false),
    onDropRejected: () => setIsDragActive(false),
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center gap-3 cursor-pointer
        transition-colors select-none
        ${isDragActive
          ? "border-accent bg-accent/10"
          : "border-border hover:border-accent/60 hover:bg-surface"}`}
    >
      <input {...getInputProps()} />
      {isDragActive ? (
        <FileAudio className="w-10 h-10 text-accent" />
      ) : (
        <UploadCloud className="w-10 h-10 text-muted" />
      )}
      <p className="text-sm text-slate-300 font-medium">
        {isDragActive ? "Drop it!" : "Drag & drop an audio file here"}
      </p>
      <p className="text-xs text-muted">or click to browse · WAV, MP3, FLAC supported</p>
    </div>
  );
}
