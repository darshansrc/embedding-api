// src/index.ts
import { Hono } from "hono";
import { pipeline, FeatureExtractionPipeline } from "@huggingface/transformers";
import { embedQueue } from "./queue";

const app = new Hono();

console.log("Loading model...");
let embedder: FeatureExtractionPipeline | undefined;

pipeline("feature-extraction", "BAAI/bge-base-en-v1.5", { dtype: "fp32" }).then(
  (model) => {
    embedder = model;
    console.log("✓ Model ready");
  },
);

app.get("/health", (c) => {
  return c.json({
    status: embedder ? "ready" : "loading",
    queue: {
      depth: embedQueue.depth,
      active: embedQueue.active,
    },
  });
});

app.post("/embed", async (c) => {
  if (!embedder) {
    return c.json({ error: "Model still loading, try again shortly" }, 503);
  }

  const { value } = await c.req.json();

  if (!value || typeof value !== "string") {
    return c.json({ error: "value (string) is required" }, 400);
  }

  const queueDepthAtArrival = embedQueue.depth;
  const t0 = performance.now();

  try {
    const result = await embedQueue.add(async () => {
      const queuedMs = parseFloat((performance.now() - t0).toFixed(2));
      const ti = performance.now();
      const output = await embedder!(value, {
        pooling: "mean",
        normalize: true,
      });
      const inferenceMs = parseFloat((performance.now() - ti).toFixed(2));
      const embedding = Array.from(output.data as Float32Array);
      return { embedding, queuedMs, inferenceMs };
    });

    const totalMs = parseFloat((performance.now() - t0).toFixed(2));

    return c.json({
      embedding: result.embedding,
      metrics: {
        totalMs,
        queuedMs: result.queuedMs,
        inferenceMs: result.inferenceMs,
        embeddingDimensions: result.embedding.length,
        queueDepthAtArrival,
        model: "BAAI/bge-base-en-v1.5",
      },
    });
  } catch (err: any) {
    if (err.message?.includes("Queue full")) {
      return c.json({ error: err.message }, 429);
    }
    throw err;
  }
});

app.post("/embed-many", async (c) => {
  if (!embedder) {
    return c.json({ error: "Model still loading, try again shortly" }, 503);
  }

  const { values } = await c.req.json();

  if (!Array.isArray(values) || values.length === 0) {
    return c.json({ error: "values (string[]) is required" }, 400);
  }

  if (values.some((v) => typeof v !== "string")) {
    return c.json({ error: "All values must be strings" }, 400);
  }

  if (values.length > 100) {
    return c.json({ error: "Maximum 100 values per request" }, 400);
  }

  const queueDepthAtArrival = embedQueue.depth;
  const t0 = performance.now();

  try {
    const result = await embedQueue.add(async () => {
      const queuedMs = parseFloat((performance.now() - t0).toFixed(2));
      const ti = performance.now();
      const outputs = await Promise.all(
        values.map((v) => embedder!(v, { pooling: "mean", normalize: true })),
      );
      const inferenceMs = parseFloat((performance.now() - ti).toFixed(2));
      const embeddings = outputs.map((o) => Array.from(o.data as Float32Array));
      return { embeddings, queuedMs, inferenceMs };
    });

    const totalMs = parseFloat((performance.now() - t0).toFixed(2));

    return c.json({
      embeddings: result.embeddings,
      metrics: {
        totalMs,
        queuedMs: result.queuedMs,
        inferenceMs: result.inferenceMs,
        count: result.embeddings.length,
        embeddingDimensions: result.embeddings[0].length,
        queueDepthAtArrival,
        model: "BAAI/bge-base-en-v1.5",
      },
    });
  } catch (err: any) {
    if (err.message?.includes("Queue full")) {
      return c.json({ error: err.message }, 429);
    }
    throw err;
  }
});

// Bun's native server — no adapter needed
export default {
  port: 3000,
  fetch: app.fetch,
};
