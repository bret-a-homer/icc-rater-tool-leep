import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from env

function buildSystemPrompt(ctx) {
  let prompt = `You are a helpful assistant embedded in the LEEP ICC Rater Agreement Calculator. You help users understand inter-rater reliability (ICC) analysis. Answer clearly for a non-technical audience such as HR professionals or hiring managers. Be concise — 2–4 sentences unless a longer answer is clearly needed. Do not use jargon.`;

  if (ctx) {
    prompt += `

The user has just run an ICC analysis with these results:
- ICC value: ${ctx.icc}
- Reliability category: ${ctx.label}
- Agreement type: ${ctx.agreementType === "absolute" ? "Absolute Agreement ICC(2,1)" : "Consistency ICC(2,1)"}
- 95% Confidence Interval: [${ctx.ciLow}, ${ctx.ciHigh}]
- Number of cases rated: ${ctx.n}
- Number of raters: ${ctx.k}
- Rating scale: 1–4 (passing cut point: scores ≥ ${ctx.cutPoint ?? 3} are passing, scores < ${ctx.cutPoint ?? 3} are rejected)

Use this context when answering questions about their results.`;
  }

  return prompt;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { messages, iccContext } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Invalid request body" });
  }

  try {
    const response = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 600,
      system: buildSystemPrompt(iccContext),
      messages,
    });

    res.json({ reply: response.content[0].text });
  } catch (e) {
    console.error("Claude API error:", e);
    const status = e?.status ?? 500;
    const msg =
      status === 401 ? "API key missing or invalid. Please check Vercel environment variables." :
      status === 429 ? "Rate limit reached. Please try again in a moment." :
      status === 400 ? `Bad request: ${e?.message ?? "unknown"}` :
      "Failed to get a response. Please try again.";
    res.status(status < 500 ? status : 500).json({ error: msg });
  }
}
