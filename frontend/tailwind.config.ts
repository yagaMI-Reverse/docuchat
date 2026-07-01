import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0B0F17",
        panel: "#111826",
        line: "#1E2735",
        brand: { DEFAULT: "#6366F1", 2: "#22D3EE" },
        ink: { DEFAULT: "#E6EAF2", dim: "#93A0B4", faint: "#5C6b80" },
      },
      fontFamily: { sans: ["Inter", "sans-serif"], mono: ["JetBrains Mono", "monospace"] },
      keyframes: { blink: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0" } } },
      animation: { blink: "blink 1s step-start infinite" },
    },
  },
  plugins: [],
} satisfies Config;
