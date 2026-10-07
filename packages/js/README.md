# unee

**Unee, from UNEEVERSE, by Muneef Mumthas.** A small open AI model that runs inside your web app or Node.js service: no server, no API key, no cost per call.

- **Decisions** with a calibrated probability for every option: yes/no, pick one, or score on a scale. It accepts the same requests as Jev's `/v1/systemone`.
- **Streaming chat** that answers from your own documents, and says when they don't cover the question.
- **Summaries** of long email, chat and call threads.

```bash
npm i unee @huggingface/transformers
```

```js
import { Unee } from "unee";

// Browser: WebGPU (recent Chrome, Edge, Safari 26+, Firefox 141+). Node.js: { device: "cpu" }.
const unee = await Unee.load("uneeverse/unee-0.8b");

const res = await unee.decide({
  state: "I was charged twice for order #4821, please refund the extra $39",
  questions: {
    team: { type: "choice", instructions: "Which team handles this?",
            criteria: { billing: "Charges and refunds", technical: "Bugs and outages", shipping: "Deliveries" } },
    refund: { type: "noul", instructions: "Does the customer ask for money back?" },
  },
});
// res.answers.team -> { choice: "billing", probabilities: { billing: 0.98, ... }, confidence: ... }

const helpCentre = [{ title: "Refunds", text: "Refunds are paid within 14 days of a return..." }];
for await (const piece of unee.stream([{ role: "user", content: "How long do refunds take?" }], { knowledge: helpCentre })) {
  chat.append(piece);
}

for await (const piece of unee.summarize(longThread, { focus: "next steps for the agent" })) summary.append(piece);
```

**Strict mode.** `unee.stream(messages, { knowledge: helpCentre, strict: true, onReport })` writes the whole answer, checks every sentence against your documents with the model's own decision side, removes what it cannot support, and yields the checked answer once. It lowers made-up facts; it does not rule them out. The measured rates are in the main repository's README.

The model downloads once (about 470 MB for Unee 0.8B) and the browser caches it.

**More:**
- Python, an HTTP server and Docker: see the main repository.
- What Unee is and how it compares: [uneeverse.net/unee](https://uneeverse.net/unee).

Apache-2.0, © Muneef Mumthas / UNEEVERSE. If you build something with Unee, please credit "Unee by UNEEVERSE".
