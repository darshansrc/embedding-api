// src/index.ts
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { pipeline, FeatureExtractionPipeline } from "@huggingface/transformers";

const app = new Hono();

// Load once on startup — warm forever
console.log("Loading model...");
let embedder: FeatureExtractionPipeline | undefined;

pipeline("feature-extraction", "BAAI/bge-base-en-v1.5", { dtype: "fp32" }).then(
  (model) => {
    embedder = model;
    console.log("✓ Model ready");
  },
);

app.get("/health", (c) => {
  return c.json({ status: embedder ? "ready" : "loading" });
});

app.post("/embed", async (c) => {
  if (!embedder) {
    return c.json({ error: "Model still loading, try again shortly" }, 503);
  }

  const { value } = await c.req.json();

  if (!value || typeof value !== "string") {
    return c.json({ error: "value (string) is required" }, 400);
  }

  const t0 = performance.now();
  const output = await embedder(value, { pooling: "mean", normalize: true });
  const inferenceMs = parseFloat((performance.now() - t0).toFixed(2));

  const embedding = Array.from(output.data as Float32Array);

  return c.json({
    embedding,
    metrics: {
      inferenceMs,
      embeddingDimensions: embedding.length,
      model: "BAAI/bge-base-en-v1.5",
    },
  });
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

  const t0 = performance.now();
  const outputs = await Promise.all(
    values.map((v) => embedder!(v, { pooling: "mean", normalize: true })),
  );
  const inferenceMs = parseFloat((performance.now() - t0).toFixed(2));

  const embeddings = outputs.map((o) => Array.from(o.data as Float32Array));

  return c.json({
    embeddings,
    metrics: {
      inferenceMs,
      count: embeddings.length,
      embeddingDimensions: embeddings[0].length,
      model: "BAAI/bge-base-en-v1.5",
    },
  });
});

serve(
  {
    fetch: app.fetch,
    port: 3000,
  },
  (info) => {
    console.log(`Server is running on http://localhost:${info.port}`);
  },
);
