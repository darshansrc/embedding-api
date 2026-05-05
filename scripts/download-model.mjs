// scripts/download-model.mjs
import { pipeline } from "@huggingface/transformers";

console.log("Downloading model...");
await pipeline("feature-extraction", "BAAI/bge-base-en-v1.5", {
  dtype: "fp32",
});
console.log("✓ Model cached");
process.exit(0);
