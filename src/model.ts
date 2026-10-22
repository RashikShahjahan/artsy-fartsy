import { createModels, createProvider, type Model } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { MODEL_ID, PROVIDER } from "./config.js";

export function liquidModels(baseUrl = process.env.ARTSY_MODEL_URL ?? "http://127.0.0.1:8080/v1") {
  const model: Model<"openai-completions"> = {
    id: MODEL_ID, name: "Liquid LFM2.5 230M", provider: PROVIDER,
    api: "openai-completions", baseUrl, reasoning: false, input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768, maxTokens: 2048,
    compat: { supportsDeveloperRole: false, supportsStore: false, maxTokensField: "max_tokens" },
  };
  const models = createModels();
  models.setProvider(createProvider({
    id: PROVIDER, baseUrl, models: [model], api: openAICompletionsApi(),
    auth: { apiKey: { name: "Local Liquid server", resolve: async () => ({ auth: { apiKey: "local-no-auth" } }) } },
  }));
  return models;
}
