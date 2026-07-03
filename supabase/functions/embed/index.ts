// Embedding service for DocuChat — runs Supabase's built-in gte-small model
// (384-dim), so vector search works with zero external API costs.
// Secured by a shared secret header; deploy with --no-verify-jwt.

const session = new Supabase.ai.Session("gte-small");

Deno.serve(async (req) => {
  if (req.headers.get("x-embed-secret") !== Deno.env.get("EMBED_SECRET")) {
    return new Response("forbidden", { status: 403 });
  }

  let texts: unknown;
  try {
    ({ texts } = await req.json());
  } catch {
    return new Response("bad json", { status: 400 });
  }
  if (!Array.isArray(texts) || texts.length === 0 || texts.length > 64) {
    return new Response("texts must be a non-empty array (max 64)", { status: 400 });
  }

  const embeddings: number[][] = [];
  for (const t of texts) {
    const vec = (await session.run(String(t).slice(0, 4000), {
      mean_pool: true,
      normalize: true,
    })) as number[];
    embeddings.push(Array.from(vec));
  }
  return Response.json({ embeddings });
});
