import { allowedRoute, trustedOrigin } from "@/lib/proxy";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
async function proxy(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const origin = request.headers.get("origin");
  if (
    request.method === "POST" &&
    !trustedOrigin(origin, request.headers.get("host"))
  ) {
    return Response.json(
      { error: "Untrusted request origin." },
      { status: 403 },
    );
  }
  const path = (await context.params).path.join("/");
  if (!allowedRoute(request.method, path))
    return Response.json({ error: "Not found." }, { status: 404 });
  const backend = process.env.ARTSY_CLIENT_API_URL || "http://127.0.0.1:3001";
  try {
    const body = request.method === "POST" ? await request.text() : undefined;
    if (body && body.length > 20000)
      return Response.json({ error: "Request is too large." }, { status: 413 });
    const response = await fetch(new URL(`/api/${path}`, backend), {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(path === "artsy/speech" ? 120000 : 30000),
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("content-type") || "application/json",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    console.error("Drawing service request failed:", error);
    return Response.json(
      {
        error:
          "Drawing is temporarily unavailable. Please try again in a moment.",
      },
      { status: 502 },
    );
  }
}
export { proxy as GET, proxy as POST };
