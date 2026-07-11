/**
 * Promptfoo custom provider for the live DocuChat RAG API.
 *
 * POSTs the test prompt to /chat, collects the SSE stream and returns
 * the final answer plus retrieval metadata (sources + grounded flag)
 * appended in a stable, assertable format:
 *
 *   <answer text>
 *   ---
 *   sources: Shipping Policy (0.923), ...
 *   grounded: true
 */
const API_BASE = process.env.DOCUCHAT_API || "https://docuchat-api-odw2.onrender.com";

class DocuChatProvider {
  constructor(options = {}) {
    this.providerId = options.id || "docuchat-prod";
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt) {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: prompt }),
    });
    if (!resp.ok) {
      return { error: `DocuChat API HTTP ${resp.status}` };
    }

    // The endpoint streams SSE; awaiting text() gives us the whole stream.
    const raw = await resp.text();
    let answer = "";
    let meta = null;
    for (const line of raw.split("\n")) {
      const l = line.trim();
      if (!l.startsWith("data: ")) continue;
      const d = JSON.parse(l.slice(6));
      if (d.delta) answer += d.delta;
      if (d.done) meta = d;
    }

    const sources =
      (meta?.sources || []).map((s) => `${s.title} (${s.score})`).join(", ") || "none";
    return {
      output: `${answer.trim()}\n---\nsources: ${sources}\ngrounded: ${meta?.grounded ?? "unknown"}`,
    };
  }
}

module.exports = DocuChatProvider;
