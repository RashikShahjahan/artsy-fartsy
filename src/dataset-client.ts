export type Message = { role: "system" | "user" | "assistant"; content: string | Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }> };
export type Completion = { text: string; model: string; usage?: { prompt_tokens: number; completion_tokens: number }; finishReason: string };
export type ChatClient = { model: string; baseUrl: string; complete(messages: Message[], maxTokens: number): Promise<Completion> };

/** The same non-streaming chat protocol used by inference/server.py. */
export function chatClient(options: { baseUrl: string; model: string; apiKey?: string; timeoutMs?: number }): ChatClient {
  const url = new URL(options.baseUrl);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error("Endpoint must be an HTTP(S) base URL without credentials, query, or fragment");
  if (!options.model.trim()) throw new Error("A model identifier is required");
  const baseUrl = url.toString().replace(/\/$/, "");
  return { model: options.model, baseUrl, async complete(messages, maxTokens) {
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST", signal: AbortSignal.timeout(options.timeoutMs ?? 60_000),
      headers: { "Content-Type": "application/json", ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}) },
      body: JSON.stringify({ model: options.model, messages, max_tokens: maxTokens, stream: false }),
    });
    // Do not persist response bodies on HTTP errors: gateways sometimes echo secrets.
    if (!response.ok) throw new Error(`Chat endpoint returned HTTP ${response.status}`);
    const body = await response.json() as any;
    const choice = body.choices?.[0];
    if (typeof choice?.message?.content !== "string") throw new Error("Chat endpoint returned no text completion");
    const usage = body.usage;
    if (usage !== undefined && (![usage?.prompt_tokens, usage?.completion_tokens].every(n => Number.isSafeInteger(n) && n >= 0))) throw new Error("Invalid usage counts from chat endpoint");
    return { text: choice.message.content, model: typeof body.model === "string" ? body.model : options.model,
      usage: usage ? { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens } : undefined,
      finishReason: choice.finish_reason ?? "unknown" };
  } };
}

export function parseJsonReply(text: string): unknown {
  const trimmed = text.trim();
  const fenced = /^```(?:json)?\s*\n([\s\S]*?)\n```$/.exec(trimmed);
  return JSON.parse(fenced ? fenced[1] : trimmed);
}
