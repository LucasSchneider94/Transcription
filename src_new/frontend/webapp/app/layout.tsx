import type { Metadata } from "next";
import { Raleway } from "next/font/google";
import "./globals.css";

const raleway = Raleway({ subsets: ["latin"], weight: ["100", "200", "300"] });

export const metadata: Metadata = {
  title: "Piano Transcription",
  description: "AI-powered piano audio transcription",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`min-h-screen bg-background text-slate-200 antialiased ${raleway.className}`}>
        {children}
      </body>
    </html>
  );
}
