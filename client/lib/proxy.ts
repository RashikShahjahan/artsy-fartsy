// Consumer routes only. The internal job launchers and review APIs stay private to ui/.
export function allowedRoute(method: string, path: string): boolean {
  if (method === "GET")
    return /^runs\/[a-f0-9]{32}(?:\/(?:image\.png|preview-\d+-\d+\.png))?$/.test(
      path,
    );
  return (
    method === "POST" &&
    (["artsy/runs", "artsy/speech"].includes(path) ||
      /^runs\/[a-f0-9]{32}\/cancel$/.test(path))
  );
}

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
