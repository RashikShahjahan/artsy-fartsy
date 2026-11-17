export function trustedOrigin(
  origin: string | null,
  host: string | null,
): boolean {
  if (!origin) return true;
  try {
    const url = new URL(origin);
    return ["http:", "https:"].includes(url.protocol) && url.host === host;
  } catch {
    return false;
  }
}
