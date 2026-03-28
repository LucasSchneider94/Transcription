import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Piano Transcription",
  description: "AI-powered piano audio transcription",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-slate-200 antialiased">
        {children}
      </body>
    </html>
  );
}
