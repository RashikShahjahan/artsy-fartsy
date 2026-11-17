import type { SVGProps } from "react";
const paths = {
  spark: "m12 2 2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7Z",
  plus: "M12 5v14M5 12h14",
  close: "m6 6 12 12M6 18 18 6",
  arrow: "M12 19V5m-6 6 6-6 6 6",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4",
  mic: "M9 5a3 3 0 0 1 6 0v7a3 3 0 0 1-6 0ZM5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8",
  sound: "m11 4-6 5H2v6h3l6 5ZM15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14",
  mute: "m11 4-6 5H2v6h3l6 5ZM16 9l6 6m0-6-6 6",
  image: "M3 3h18v18H3ZM3 16l5-5 5 5 3-3 5 5M15 8h.01",
  history: "M3 11a9 9 0 1 1 2.6 7.4M3 4v7h7m2-4v5l3 2",
  keyboard:
    "M3 5h18v14H3ZM6 9h1m3 0h1m3 0h1m3 0h1M6 12h1m3 0h1m3 0h1m3 0h1M7 15h10",
};
export function Icon({
  name,
  ...props
}: SVGProps<SVGSVGElement> & { name: keyof typeof paths }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d={paths[name]} />
    </svg>
  );
}
