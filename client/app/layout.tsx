import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "Artsy — Let’s draw",
  description:
    "A little imagination, a little conversation. Tell Artsy what to draw.",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
