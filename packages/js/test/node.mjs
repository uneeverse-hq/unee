// Node smoke test: decisions + streaming chat on CPU (onnxruntime-node).
//   node packages/js/test/node.mjs <model dir containing onnx/ and tokenizer files>
import { env } from "@huggingface/transformers";
import { Unee } from "../src/index.js";
import path from "node:path";

const dir = path.resolve(process.argv[2] || "models/onnx-unee-0.8b");
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = path.dirname(dir) + path.sep;
const t0 = performance.now();
const unee = await Unee.load(path.basename(dir), { device: "cpu", dtype: "q4f16" });
console.log(`loaded in ${Math.round(performance.now() - t0)} ms`);
const res = await unee.decide({
  state: "Hi, I was charged twice for order #4821 this morning. Please send the extra $39 back to my card.",
  questions: {
    team: { type: "choice", instructions: "Which team should handle this ticket?",
            criteria: { billing: "Charges, refunds and payments", technical: "Bugs and outages", shipping: "Deliveries" } },
    refund: { type: "noul", instructions: "Does the customer ask for money back?" },
  },
});
console.log(JSON.stringify(res));
let text = "";
for await (const piece of unee.stream([{ role: "user", content: "Say hello in five words." }], { max_new_tokens: 20, temperature: 0 })) text += piece;
console.log("stream:", text);
