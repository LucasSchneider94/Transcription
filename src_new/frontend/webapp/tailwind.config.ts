import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // All values mirror the CSS custom properties in globals.css.
        // Change the palette there; Tailwind picks it up via var().
        background:      "var(--color-bg)",
        surface:         "var(--color-surface)",
        surface2:        "var(--color-surface2)",
        border:          "var(--color-border)",
        "border-strong": "var(--color-border-strong)",
        accent:          "var(--color-accent)",
        "accent-light":  "var(--color-accent-light)",
        "accent-dim":    "var(--color-accent-dim)",
        muted:           "var(--color-muted)",
      },
    },
  },
  plugins: [],
};

export default config;
