export type ArtworkFailure = "missing" | "connection" | "invalid";

export function artworkFailure(status: number): ArtworkFailure {
  if (status === 404 || status === 410) return "missing";
  if (status >= 500) return "connection";
  return "invalid";
}
