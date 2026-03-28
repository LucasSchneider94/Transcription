import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0f0f13",
        surface: "#1a1a24",
        border: "#2a2a3a",
        accent: "#7c6af7",
        "accent-light": "#a89df9",
        muted: "#6b7280",
      },
    },
  },
  plugins: [],
};

export default config;
