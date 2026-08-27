import { cn } from "./ui/utils";

export interface LogoProps {
  /** Size in Tailwind size-* notation, e.g. "size-8", "size-12", "size-16" */
  className?: string;
  /** Show "InsightAI" wordmark to the right of the icon. */
  withWordmark?: boolean;
  /** Wordmark size relative to icon (default 1.0). */
  wordmarkScale?: number;
}

/**
 * InsightAI brand logo.
 *
 * Visual identity: a square with rounded corners, filled with a diagonal
 * gradient from slate-900 (top-left) to sky-700 (bottom-right). A bold,
 * white "V" sits centered — the user's chosen mark — in a clean geometric
 * form that reads at 16x16 (favicon) and scales up to 256x256 (marketing).
 *
 * The wordmark is set in Inter/Manrope-style sans, kerned tight, with the
 * "V" in the same sky gradient to echo the icon. Everything is SVG so it
 * stays crisp at any DPR and respects prefers-color-scheme.
 */
export function Logo({
  className = "size-12",
  withWordmark = false,
  wordmarkScale = 1.0,
}: LogoProps) {
  // The viewBox is 32x32. Tailwind size-* sets the rendered box; we let
  // the browser scale the SVG. The inner strokes are sized to read well
  // at 16px (favicon) and stay balanced at 64px+.
  const wordmarkStyle = withWordmark
    ? { fontSize: `${1 * wordmarkScale}rem` }
    : undefined;

  return (
    <span
      className={cn("inline-flex items-center gap-2.5", className)}
      aria-label="InsightAI"
    >
      <svg
        viewBox="0 0 32 32"
        xmlns="http://www.w3.org/2000/svg"
        className="h-full w-auto shrink-0"
        role="img"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="ia-bg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0f172a" />
            <stop offset="100%" stopColor="#0369a1" />
          </linearGradient>
          <linearGradient id="ia-sheen" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="ia-v" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="100%" stopColor="#0369a1" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="32" height="32" rx="7" ry="7" fill="url(#ia-bg)" />
        <rect
          x="0"
          y="0"
          width="32"
          height="16"
          rx="7"
          ry="7"
          fill="url(#ia-sheen)"
        />
        {/* Letter V — two diagonal strokes converging at the bottom */}
        <path
          d="M 8 8 L 11 8 L 16 19.5 L 21 8 L 24 8 L 17.2 24 L 14.8 24 Z"
          fill="#ffffff"
          stroke="#ffffff"
          strokeWidth="0.6"
          strokeLinejoin="round"
        />
      </svg>
      {withWordmark && (
        <span
          className="font-semibold tracking-tight text-zinc-950"
          style={wordmarkStyle}
        >
          <span className="bg-gradient-to-r from-sky-600 to-sky-800 bg-clip-text text-transparent">
            V
          </span>
          InsightAI
        </span>
      )}
    </span>
  );
}
