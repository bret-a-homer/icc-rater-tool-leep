import { useState, useCallback, useRef, useEffect } from "react";

// ── ICC calculation (pure JS, no external deps) ─────────────────────────────
function calculateICC(matrix, type) {
  // matrix: array of rows, each row is array of ratings
  // type: "absolute" → ICC(2,1) two-way random, absolute agreement
  //       "relative"  → ICC(2,1) two-way random, consistency
  const n = matrix.length;       // subjects (cases)
  const k = matrix[0].length;    // raters

  // Grand mean
  let grandSum = 0;
  for (const row of matrix) for (const v of row) grandSum += v;
  const grandMean = grandSum / (n * k);

  // Row means (subject means)
  const rowMeans = matrix.map(row => row.reduce((a, b) => a + b, 0) / k);

  // Col means (rater means)
  const colMeans = Array(k).fill(0);
  for (const row of matrix) row.forEach((v, j) => { colMeans[j] += v; });
  colMeans.forEach((_, j) => { colMeans[j] /= n; });

  // SS_between-subjects (rows)
  let SSR = 0;
  for (let i = 0; i < n; i++) SSR += k * Math.pow(rowMeans[i] - grandMean, 2);

  // SS_between-raters (cols)
  let SSC = 0;
  for (let j = 0; j < k; j++) SSC += n * Math.pow(colMeans[j] - grandMean, 2);

  // SS_total
  let SST = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < k; j++)
      SST += Math.pow(matrix[i][j] - grandMean, 2);

  // SS_error (residual)
  const SSE = SST - SSR - SSC;

  const dfR = n - 1;
  const dfC = k - 1;
  const dfE = (n - 1) * (k - 1);

  const MSR = SSR / dfR;
  const MSC = SSC / dfC;
  const MSE = SSE / dfE;

  let icc;
  if (type === "absolute") {
    // ICC(2,1) Absolute Agreement
    icc = (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE));
  } else {
    // ICC(2,1) Consistency
    icc = (MSR - MSE) / (MSR + (k - 1) * MSE);
  }

  // 95% CI via F-distribution approximation
  const F = MSR / MSE;
  const FL = F / 3.84; // rough 95% lower (F_crit ≈ 3.84 for large df)
  const FU = F * 3.84;
  const ciLow = Math.max(-1, (FL - 1) / (FL + k - 1));
  const ciHigh = Math.min(1, (FU - 1) / (FU + k - 1));

  return { icc: Math.max(-1, Math.min(1, icc)), ciLow, ciHigh, n, k };
}

function parseCSV(text) {
  const lines = text.trim().split("\n").filter(l => l.trim());
  const hasHeader = isNaN(parseFloat(lines[0].split(/[,\t]/)[0]));
  const dataLines = hasHeader ? lines.slice(1) : lines;
  const matrix = dataLines.map(line =>
    line.split(/[,\t]/).map(v => parseFloat(v.trim()))
  ).filter(row => row.every(v => !isNaN(v)));
  return { matrix, headers: hasHeader ? lines[0].split(/[,\t]/).map(s => s.trim()) : null };
}

// ── Normal CDF (for misclassification math) ──────────────────────────────────
function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}
function normalCDF(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

// ── Simulated data generator ─────────────────────────────────────────────────
function generateSimulatedData(targetICC, n = 40, k = 3) {
  // errorSD derived from ICC = σ²_subjects / (σ²_subjects + σ²_error)
  // for uniform true scores on [1,4]: σ²_subjects = (3²)/12 = 0.75
  const errorSD = Math.sqrt(0.75 * (1 - targetICC) / targetICC);
  const matrix = [];
  for (let i = 0; i < n; i++) {
    const trueScore = 1 + Math.random() * 3;
    const row = [];
    for (let j = 0; j < k; j++) {
      // Box-Muller normal sample
      const u1 = Math.max(1e-10, Math.random());
      const noise = errorSD * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
      row.push(Math.min(4, Math.max(1, Math.round(trueScore + noise))));
    }
    matrix.push(row);
  }
  return matrix;
}

// ── Misclassification Panel ───────────────────────────────────────────────────
function MisclassificationPanel({ icc, cutPoint = 3 }) {
  const TOTAL = 1000;
  // Uniform on [1,4]: count cases in each "true" band around cutPoint
  // Band just above cut: [cutPoint, cutPoint+1); band well above: [cutPoint+1, 4]
  const bandSize = 0.25; // each integer band = 25% of uniform [1,4]
  const trueAtCut  = TOTAL * bandSize; // e.g. true score = cutPoint
  const trueAbove  = TOTAL * bandSize; // e.g. true score = cutPoint + 1

  const scaleVariance = (3 * 3) / 12; // uniform on [1,4]
  const errorSD = Math.sqrt((1 - Math.max(0, Math.min(0.9999, icc))) * scaleVariance);

  // Continuous threshold at (cutPoint - 0.5): P(rated < cutPoint | true = x)
  const thresh = cutPoint - 0.5;
  const pAtCut  = normalCDF((thresh - cutPoint) / errorSD);
  const pAbove  = normalCDF((thresh - (cutPoint + 1)) / errorSD);

  const falseFromCut  = Math.round(trueAtCut  * pAtCut);
  const falseFromAbove = Math.round(trueAbove * pAbove);
  const totalFalse = falseFromCut + falseFromAbove;
  const qualifiedCount = trueAtCut + trueAbove;
  const pct = ((totalFalse / qualifiedCount) * 100).toFixed(1);

  const barW = Math.min(100, parseFloat(pct));
  const barColor = parseFloat(pct) < 5 ? "#27ae60" : parseFloat(pct) < 15 ? "#f39c12" : "#e74c3c";

  return (
    <div style={{
      background: "#141720",
      borderRadius: "14px",
      padding: "1.25rem 1.5rem",
      border: "1px solid #2a2d3e",
      marginTop: "1rem",
    }}>
      <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "#888",
                    textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.85rem" }}>
        False Rejection Estimator &nbsp;·&nbsp; 1,000 applicants assumed
      </div>

      {/* Big number */}
      <div style={{ display: "flex", alignItems: "flex-end", gap: "0.75rem", marginBottom: "0.85rem" }}>
        <div style={{ fontSize: "2.8rem", fontWeight: 700, lineHeight: 1,
                      fontFamily: "'DM Mono', monospace", color: barColor }}>
          {totalFalse}
        </div>
        <div style={{ fontSize: "0.85rem", color: "#888", paddingBottom: "0.4rem", lineHeight: 1.4 }}>
          truly-qualified applicants (true score ≥ {cutPoint})<br />
          <strong style={{ color: "#ccc" }}>incorrectly rejected (rated &lt; {cutPoint})</strong>
        </div>
      </div>

      {/* Bar */}
      <div style={{ height: "8px", borderRadius: "4px", background: "#2a2d3e",
                    marginBottom: "0.5rem", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${barW}%`, background: barColor,
                      borderRadius: "4px", transition: "width 0.4s" }} />
      </div>
      <div style={{ fontSize: "0.72rem", color: "#555", marginBottom: "1rem" }}>
        {pct}% of the {Math.round(qualifiedCount)} truly-qualified applicants (true score ≥ {cutPoint}) are rated &lt; {cutPoint} and rejected
      </div>

      {/* Breakdown table */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem" }}>
        {[
          { label: `True score ${cutPoint}, rated < ${cutPoint}`, count: falseFromCut,  pct: (pAtCut  * 100).toFixed(1), note: "Borderline-pass rated as fail" },
          { label: `True score ${cutPoint + 1}, rated < ${cutPoint}`, count: falseFromAbove, pct: (pAbove * 100).toFixed(1), note: "Clear-pass rated as fail" },
        ].map(({ label, count, pct: p, note }) => (
          <div key={label} style={{ background: "#1a1d27", borderRadius: "8px",
                                    padding: "0.75rem", border: "1px solid #2a2d3e" }}>
            <div style={{ fontSize: "1.4rem", fontWeight: 700, color: barColor,
                          fontFamily: "'DM Mono', monospace" }}>{count}</div>
            <div style={{ fontSize: "0.7rem", color: "#888", marginTop: "0.2rem" }}>{note}</div>
            <div style={{ fontSize: "0.65rem", color: "#555", marginTop: "0.15rem" }}>
              {p}% chance per applicant
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: "0.85rem", fontSize: "0.7rem", color: "#555", lineHeight: 1.8 }}>
        <div style={{ fontWeight: 700, color: "#666", marginBottom: "0.3rem", textTransform: "uppercase",
                      letterSpacing: "0.06em", fontSize: "0.65rem" }}>Assumptions</div>
        <div>1. Each case is rated by 1 rater</div>
        <div>2. Cases rated {cutPoint} or higher are passing</div>
        <div>3. Cases rated below {cutPoint} are rejected</div>
      </div>
    </div>
  );
}

// ── Per-Rater Breakdown ───────────────────────────────────────────────────────
function PerRaterBreakdown({ matrix, headers, cutPoint = 3 }) {
  if (!matrix || matrix.length < 2 || matrix[0].length < 2) return null;

  const k = matrix[0].length;
  const n = matrix.length;

  // Estimated true score per case = mean across all raters
  const rowMeans = matrix.map(row => row.reduce((a, b) => a + b, 0) / row.length);

  // "True" pass/fail: mean >= cutPoint → pass
  const truePass = i => rowMeans[i] >= cutPoint;

  const raters = Array.from({ length: k }, (_, j) => {
    const name = headers?.[j] || `Rater ${j + 1}`;

    // Confusion matrix counts
    let tp = 0, fp = 0, tn = 0, fn = 0;
    // Distribution counts for scores 1–4
    const raterDist  = [0, 0, 0, 0]; // index = score-1
    const trueDist   = [0, 0, 0, 0]; // rounded mean

    matrix.forEach((row, i) => {
      const score = row[j];
      const raterPass = score >= cutPoint;
      const tp_ = truePass(i);
      if (tp_ && raterPass)  tp++;
      if (!tp_ && raterPass) fp++;
      if (!tp_ && !raterPass) tn++;
      if (tp_ && !raterPass) fn++;

      raterDist[Math.round(score) - 1] = (raterDist[Math.round(score) - 1] || 0) + 1;
      const roundedMean = Math.min(4, Math.max(1, Math.round(rowMeans[i])));
      trueDist[roundedMean - 1]++;
    });

    const precision = tp + fp > 0 ? tp / (tp + fp) : null;
    const recall    = tp + fn > 0 ? tp / (tp + fn) : null;

    return { name, tp, fp, tn, fn, precision, recall, raterDist, trueDist };
  });

  // Smooth overlapping curve chart for distributions
  const DistCurve = ({ raterDist, trueDist, cut }) => {
    const W = 180, H = 80;
    const PAD = { l: 10, r: 10, t: 10, b: 20 };
    const cW = W - PAD.l - PAD.r;
    const cH = H - PAD.t - PAD.b;

    const total = raterDist.reduce((a, b) => a + b, 0) || 1;
    const rP = raterDist.map(c => c / total);
    const tP = trueDist.map(c => c / total);
    const maxP = Math.max(0.01, ...rP, ...tP);

    // Map score (1–4) to x, proportion to y
    const xOf = s => PAD.l + ((s - 1) / 3) * cW;
    const yOf = p => PAD.t + cH - (p / maxP) * cH;
    const baseY = PAD.t + cH;

    // Catmull-Rom → cubic Bezier smooth path through 4 points
    const smoothPath = props => {
      const pts = [1, 2, 3, 4].map((s, i) => [xOf(s), yOf(props[i])]);
      let d = `M ${pts[0][0]},${pts[0][1]}`;
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[Math.max(0, i - 1)];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[Math.min(pts.length - 1, i + 2)];
        const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
        const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
        const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
        const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
        d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
      }
      return d;
    };
    const filledPath = props =>
      `${smoothPath(props)} L ${xOf(4)},${baseY} L ${xOf(1)},${baseY} Z`;

    const threshX = xOf(cut);

    return (
      <div>
        <svg width={W} height={H} style={{ display: "block", overflow: "visible" }}>
          {/* Pass/fail threshold */}
          <line x1={threshX} y1={PAD.t - 4} x2={threshX} y2={baseY}
                stroke="#3a3f55" strokeWidth="1.5" strokeDasharray="3,2" />
          <text x={threshX + 3} y={PAD.t + 1} fontSize="6" fill="#444" dominantBaseline="hanging">cut</text>

          {/* True distribution — blue */}
          <path d={filledPath(tP)} fill="rgba(58,123,213,0.18)" />
          <path d={smoothPath(tP)} fill="none" stroke="rgba(58,123,213,0.75)" strokeWidth="1.5" strokeLinejoin="round" />

          {/* Rater distribution — orange */}
          <path d={filledPath(rP)} fill="rgba(230,126,34,0.18)" />
          <path d={smoothPath(rP)} fill="none" stroke="rgba(230,126,34,0.9)" strokeWidth="1.5" strokeLinejoin="round" />

          {/* X-axis score labels */}
          {[1, 2, 3, 4].map(s => (
            <text key={s} x={xOf(s)} y={H - 5} textAnchor="middle"
                  fontSize="7.5" fill={s >= cut ? "#27ae60" : "#c0392b"}
                  fontWeight="700" fontFamily="'DM Mono', monospace">{s}</text>
          ))}
        </svg>
        {/* Legend */}
        <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.25rem" }}>
          {[
            { color: "rgba(58,123,213,0.75)",  label: "True (mean)" },
            { color: "rgba(230,126,34,0.9)",   label: "This rater"  },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
              <svg width="16" height="8">
                <line x1="0" y1="4" x2="16" y2="4" stroke={color} strokeWidth="1.5" />
              </svg>
              <span style={{ fontSize: "0.58rem", color: "#555" }}>{label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Confusion matrix cell
  const Cell = ({ count, label, sub, bg, textColor = "#fff" }) => (
    <div style={{
      background: bg, borderRadius: "6px", padding: "0.5rem 0.4rem",
      textAlign: "center", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: "2px",
    }}>
      <div style={{ fontSize: "1.3rem", fontWeight: 700, fontFamily: "'DM Mono', monospace", color: textColor, lineHeight: 1 }}>
        {count}
      </div>
      <div style={{ fontSize: "0.6rem", fontWeight: 700, color: textColor, opacity: 0.85 }}>{label}</div>
      <div style={{ fontSize: "0.55rem", color: textColor, opacity: 0.55 }}>{sub}</div>
    </div>
  );

  return (
    <div style={{
      background: "#141720", borderRadius: "14px",
      padding: "1.25rem 1.5rem", border: "1px solid #2a2d3e", marginTop: "1rem",
    }}>
      <div style={{
        fontSize: "0.75rem", fontWeight: 700, color: "#888",
        textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.15rem",
      }}>
        Per-Rater Analysis
      </div>
      <div style={{ fontSize: "0.72rem", color: "#555", lineHeight: 1.6, marginBottom: "1.25rem" }}>
        Each rater is compared against the group consensus (mean score across all raters) using a pass/fail threshold of {cutPoint}.
        The <strong style={{ color: "#ccc" }}>confusion matrix</strong> shows how often this rater agreed or disagreed with the group:
        TP = correctly passed, TN = correctly failed, FN = falsely rejected a likely-pass candidate, FP = falsely passed a likely-fail candidate.
        The <strong style={{ color: "#1e90ff" }}>blue curve</strong> shows the true (consensus) score distribution; the <strong style={{ color: "#e67e22" }}>orange curve</strong> shows this rater's distribution.
        A large gap between the two curves suggests this rater scores systematically higher or lower than the group. n = {n} cases.
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        {raters.map(({ name, tp, fp, tn, fn, precision, recall, raterDist, trueDist }) => (
          <div key={name} style={{
            background: "#1a1d27", borderRadius: "10px",
            padding: "1rem", border: "1px solid #2a2d3e",
          }}>
            {/* Rater name */}
            <div style={{
              fontSize: "0.8rem", fontWeight: 700, color: "#ddd",
              marginBottom: "0.85rem", fontFamily: "'DM Sans', sans-serif",
            }}>{name}</div>

            {/* Confusion matrix + dist chart side by side */}
            <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start", flexWrap: "wrap" }}>

              {/* Confusion matrix */}
              <div style={{ flex: "0 0 auto" }}>
                {/* Column headers */}
                <div style={{ display: "grid", gridTemplateColumns: "56px 1fr 1fr", gap: "4px", marginBottom: "4px" }}>
                  <div />
                  <div style={{ fontSize: "0.6rem", color: "#555", textAlign: "center" }}>Rater: Pass</div>
                  <div style={{ fontSize: "0.6rem", color: "#555", textAlign: "center" }}>Rater: Fail</div>
                </div>
                {/* Row: True Pass */}
                <div style={{ display: "grid", gridTemplateColumns: "56px 1fr 1fr", gap: "4px", marginBottom: "4px" }}>
                  <div style={{ fontSize: "0.6rem", color: "#555", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: "4px" }}>
                    True: Pass
                  </div>
                  <Cell count={tp} label="TP" sub="Correct pass" bg="#1e3a2f" textColor="#2ecc71" />
                  <Cell count={fn} label="FN" sub="False reject" bg="#3a1e1e" textColor="#e74c3c" />
                </div>
                {/* Row: True Fail */}
                <div style={{ display: "grid", gridTemplateColumns: "56px 1fr 1fr", gap: "4px" }}>
                  <div style={{ fontSize: "0.6rem", color: "#555", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: "4px" }}>
                    True: Fail
                  </div>
                  <Cell count={fp} label="FP" sub="False pass" bg="#3a2a1e" textColor="#f39c12" />
                  <Cell count={tn} label="TN" sub="Correct fail" bg="#1e1e2a" textColor="#7f8c8d" />
                </div>
                {/* Precision / Recall */}
                <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.6rem" }}>
                  {[
                    { label: "Precision", val: precision },
                    { label: "Recall",    val: recall },
                  ].map(({ label, val }) => (
                    <div key={label} style={{ fontSize: "0.65rem", color: "#555" }}>
                      {label}:{" "}
                      <span style={{ color: "#aaa", fontFamily: "'DM Mono', monospace" }}>
                        {val !== null ? (val * 100).toFixed(0) + "%" : "—"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Distribution chart */}
              <div style={{ flex: 1, minWidth: "100px" }}>
                <div style={{ fontSize: "0.65rem", color: "#555", marginBottom: "0.25rem" }}>
                  Score distribution
                </div>
                <DistCurve raterDist={raterDist} trueDist={trueDist} cut={cutPoint} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: "0.85rem", fontSize: "0.6rem", color: "#444", lineHeight: 1.7 }}>
        True label is estimated as the rounded mean rating across all raters for each case.
        TP = true positive (correctly passed), FN = false negative (falsely rejected),
        FP = false positive (falsely passed), TN = true negative (correctly rejected).
      </div>
    </div>
  );
}

// ── Chat Widget ──────────────────────────────────────────────────────────────
function ChatWidget({ iccContext }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    if (endRef.current) endRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { role: "user", content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages, iccContext }),
      });
      const data = await res.json();
      setMessages(m => [...m, { role: "assistant", content: data.reply || data.error || "Something went wrong." }]);
    } catch {
      setMessages(m => [...m, { role: "assistant", content: "Couldn't connect. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: "1.5rem", borderTop: "1px solid #2a2d3e", paddingTop: "1rem" }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", background: "none", border: "none", color: "#666",
          cursor: "pointer", fontSize: "0.8rem", fontWeight: 600,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          fontFamily: "'DM Sans', sans-serif", padding: "0.25rem 0",
        }}
      >
        <span>💬 Ask a question about this analysis</span>
        <span style={{ fontSize: "0.65rem" }}>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div style={{ marginTop: "0.75rem" }}>
          {/* Chat history */}
          <div style={{
            maxHeight: "300px", overflowY: "auto",
            background: "#0f1117", borderRadius: "10px",
            padding: "0.75rem", marginBottom: "0.5rem",
            border: "1px solid #2a2d3e",
          }}>
            {messages.length === 0 && (
              <div style={{ color: "#444", fontSize: "0.75rem", textAlign: "center", padding: "1.5rem 0", lineHeight: 1.6 }}>
                Ask anything about ICC, rater agreement,<br />or how to interpret your results.
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} style={{
                marginBottom: "0.6rem", display: "flex",
                justifyContent: m.role === "user" ? "flex-end" : "flex-start",
              }}>
                <div style={{
                  background: m.role === "user" ? "rgba(245,166,35,0.1)" : "#1a1d27",
                  border: `1px solid ${m.role === "user" ? "rgba(245,166,35,0.3)" : "#2a2d3e"}`,
                  borderRadius: "8px", padding: "0.5rem 0.75rem",
                  maxWidth: "88%", fontSize: "0.8rem",
                  color: m.role === "user" ? "#f5a623" : "#ccc",
                  lineHeight: 1.6, whiteSpace: "pre-wrap",
                }}>
                  {m.content}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ color: "#555", fontSize: "0.75rem", padding: "0.25rem 0.5rem" }}>
                Thinking…
              </div>
            )}
            <div ref={endRef} />
          </div>

          {/* Input row */}
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
              placeholder="Ask a question…"
              style={{
                flex: 1, background: "#141720", border: "1px solid #2a2d3e",
                borderRadius: "8px", color: "#ccc",
                fontFamily: "'DM Sans', sans-serif", fontSize: "0.8rem",
                padding: "0.5rem 0.75rem", outline: "none",
              }}
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              style={{
                background: loading || !input.trim() ? "#2a2d3e" : "#f5a623",
                color: loading || !input.trim() ? "#555" : "#0f1117",
                border: "none", borderRadius: "8px",
                padding: "0.5rem 1rem", cursor: loading ? "not-allowed" : "pointer",
                fontFamily: "'DM Sans', sans-serif", fontSize: "0.8rem", fontWeight: 700,
                transition: "all 0.15s",
              }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Gauge / Scale component ──────────────────────────────────────────────────
function ICCGauge({ icc, ciLow, ciHigh }) {
  // Layout: negative sliver = 8% of bar width, positive range = 92%
  const NEG_PCT = 8;
  const POS_PCT = 100 - NEG_PCT;

  // Positive zones in ICC space (0..1) with their share of the positive bar
  const posZones = [
    { from: 0.000, to: 0.500, label: "Poor",               color: "#95a5a6" },
    { from: 0.500, to: 0.600, label: "Borderline",         color: "#e74c3c" },
    { from: 0.600, to: 0.750, label: "Moderate",           color: "#f1c40f" },
    { from: 0.750, to: 0.900, label: "Good",               color: "#27ae60" },
    { from: 0.900, to: 0.975, label: "Excellent",          color: "#1e90ff" },
    { from: 0.975, to: 1.000, label: "Suspiciously Ideal", color: "#95a5a6" },
  ];

  // Map an ICC value to bar % position
  const toBarPct = (v) => {
    const clamped = Math.max(-1, Math.min(1, v));
    if (clamped < 0) {
      // Negative: compress entire -1..0 range into NEG_PCT sliver
      return ((clamped + 1) / 1) * NEG_PCT;
    } else {
      // Positive: map 0..1 into NEG_PCT..100
      return NEG_PCT + clamped * POS_PCT;
    }
  };

  const allZones = [
    { from: -1, to: 0, label: "Neg.", color: "#95a5a6", widthPct: NEG_PCT },
    ...posZones.map(z => ({
      ...z,
      widthPct: (z.to - z.from) * POS_PCT,
    })),
  ];

  const getZone = (v) => {
    if (v < 0) return allZones[0];
    for (const z of posZones) if (v >= z.from && v < z.to) return z;
    return posZones[posZones.length - 1];
  };

  const zone = getZone(icc);
  const needlePct = toBarPct(icc);
  const ciLowPct  = toBarPct(Math.max(-1, ciLow));
  const ciHighPct = toBarPct(Math.min(1,  ciHigh));

  // Axis tick marks: positions in bar % and their labels
  const ticks = [
    { pct: NEG_PCT,                          label: "0.0" },
    { pct: NEG_PCT + 0.50 * POS_PCT,         label: "0.50" },
    { pct: NEG_PCT + 0.60 * POS_PCT,         label: "0.60" },
    { pct: NEG_PCT + 0.75 * POS_PCT,         label: "0.75" },
    { pct: NEG_PCT + 0.90 * POS_PCT,         label: "0.90" },
    { pct: NEG_PCT + 0.975 * POS_PCT,        label: "0.975" },
    { pct: 100,                               label: "1.0" },
  ];

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif" }}>
      {/* ICC value badge */}
      <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
        <div style={{
          display: "inline-block",
          background: zone.color,
          color: "#fff",
          borderRadius: "12px",
          padding: "1rem 2.5rem",
          boxShadow: `0 4px 24px ${zone.color}55`
        }}>
          <div style={{ fontSize: "3rem", fontWeight: "700", lineHeight: 1,
                        fontFamily: "'DM Mono', monospace" }}>
            {icc.toFixed(3)}
          </div>
          <div style={{ fontSize: "0.85rem", marginTop: "0.3rem", opacity: 0.9, fontWeight: 600 }}>
            {zone.label}
          </div>
        </div>
        <div style={{ marginTop: "0.6rem", fontSize: "0.8rem", color: "#888" }}>
          95% CI: [{ciLow.toFixed(3)}, {ciHigh.toFixed(3)}]
        </div>
      </div>

      {/* Bar */}
      <div style={{ position: "relative", height: "36px", borderRadius: "8px",
                    overflow: "hidden", display: "flex",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.15)" }}>
        {allZones.map((z, i) => (
          <div key={i} style={{
            width: `${z.widthPct}%`,
            background: z.color,
            height: "100%",
            opacity: 0.88,
            flexShrink: 0,
          }} />
        ))}

        {/* CI band — only render if spans more than 1px */}
        {ciHighPct - ciLowPct > 0.5 && (
          <div style={{
            position: "absolute",
            left: `${ciLowPct}%`,
            width: `${ciHighPct - ciLowPct}%`,
            top: 0, bottom: 0,
            background: "rgba(255,255,255,0.25)",
            borderLeft: "2px dashed rgba(255,255,255,0.85)",
            borderRight: "2px dashed rgba(255,255,255,0.85)",
          }} />
        )}

        {/* Needle */}
        <div style={{
          position: "absolute",
          left: `${needlePct}%`,
          top: "-4px", bottom: "-4px",
          width: "4px",
          background: "#fff",
          borderRadius: "2px",
          boxShadow: "0 0 8px rgba(0,0,0,0.5)",
          transform: "translateX(-50%)",
        }} />
      </div>

      {/* Zone labels row — suppress "Suspiciously Ideal" label, keep the gray bar */}
      <div style={{ display: "flex", marginTop: "5px" }}>
        {allZones.map((z, i) => (
          <div key={i} style={{
            width: `${z.widthPct}%`,
            fontSize: i === 0 ? "0.52rem" : "0.58rem",
            color: z.color,
            textAlign: "center",
            fontWeight: 600,
            lineHeight: 1.2,
            overflow: "hidden",
            whiteSpace: i === 0 ? "nowrap" : "normal",
          }}>
            {z.label === "Suspiciously Ideal" ? "" : z.label}
          </div>
        ))}
      </div>

      {/* Tick marks row — positioned absolutely over a relative container */}
      <div style={{ position: "relative", height: "20px", marginTop: "2px" }}>
        {ticks.map((t, i) => (
          <div key={i} style={{
            position: "absolute",
            left: `${t.pct}%`,
            transform: "translateX(-50%)",
            fontSize: "0.62rem",
            color: "#666",
            fontFamily: "'DM Mono', monospace",
            whiteSpace: "nowrap",
          }}>
            {t.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Noise Visualization component ────────────────────────────────────────────
function NoiseVisualization({ icc, scaleMin = 1, scaleMax = 4 }) {
  const [trueScore, setTrueScore] = useState(2.5);
  // Which comparison curves are toggled on
  const [shown, setShown] = useState({ actual: true, borderline: true, moderate: true, good: true, excellent: true });

  const scaleRange = scaleMax - scaleMin;
  const scaleVariance = (scaleRange * scaleRange) / 12;

  const errorSD = (iccVal) => {
    const c = Math.max(0, Math.min(0.9999, iccVal));
    return Math.sqrt((1 - c) * scaleVariance);
  };

  // Reference ICC values = midpoint of each category
  const comparisons = [
    { key: "excellent",   label: "Excellent",   icc: 0.925, color: "#1e90ff", dash: "6,3" },
    { key: "good",        label: "Good",        icc: 0.825, color: "#27ae60", dash: "4,4" },
    { key: "moderate",    label: "Moderate",    icc: 0.675, color: "#f1c40f", dash: "4,4" },
    { key: "borderline",  label: "Borderline",  icc: 0.550, color: "#e74c3c", dash: "2,3" },
  ];

  const getActualColor = (v) => {
    if (v < 0) return "#95a5a6";
    if (v < 0.50) return "#95a5a6";
    if (v < 0.60) return "#e74c3c";
    if (v < 0.75) return "#f1c40f";
    if (v < 0.90) return "#27ae60";
    if (v < 0.975) return "#1e90ff";
    return "#95a5a6";
  };
  const actualColor = getActualColor(icc);

  // Gaussian PDF
  const gaussian = (x, mean, sd) =>
    sd < 0.001
      ? (Math.abs(x - mean) < 0.01 ? 999 : 0)
      : (1 / (sd * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / sd, 2));

  const steps = 300;
  const xs = Array.from({ length: steps + 1 }, (_, i) => scaleMin + (i / steps) * scaleRange);

  // Compute all curves, find global max density for shared y-scale
  const allCurves = [
    { key: "actual", iccVal: icc, color: actualColor, dash: null, label: `Your ICC (${icc.toFixed(3)})`, strokeW: 3 },
    ...comparisons.map(c => ({ ...c, iccVal: c.icc, strokeW: 1.5 })),
  ];

  let globalMax = 0;
  const curvePoints = allCurves.map(c => {
    const sd = errorSD(c.iccVal);
    const pts = xs.map(x => ({ x, y: gaussian(x, trueScore, sd) }));
    const peak = Math.max(...pts.map(p => p.y));
    if (peak > globalMax) globalMax = peak;
    return { ...c, pts, sd };
  });

  const W = 460, H = 130, PAD_L = 8, PAD_R = 8, PAD_T = 14, PAD_B = 28;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const toX = x => PAD_L + ((x - scaleMin) / scaleRange) * plotW;
  const toY = y => PAD_T + plotH - (globalMax > 0 ? (y / globalMax) * plotH : 0);

  const makePath = (pts) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`).join(" ");

  const makeFill = (pts) =>
    makePath(pts) +
    ` L${toX(pts[pts.length - 1].x)},${toY(0)} L${toX(pts[0].x)},${toY(0)} Z`;

  const tickScores = [1, 1.5, 2, 2.5, 3, 3.5, 4];

  const toggle = (key) => setShown(s => ({ ...s, [key]: !s[key] }));

  return (
    <div style={{
      background: "#141720",
      borderRadius: "14px",
      padding: "1.25rem 1.5rem",
      border: "1px solid #2a2d3e",
      marginBottom: "1rem",
    }}>
      <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "#888",
                    textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.4rem" }}>
        Expected Rating Distribution
      </div>
      <div style={{ fontSize: "0.72rem", color: "#555", lineHeight: 1.6, marginBottom: "0.85rem" }}>
        Each curve shows the spread of scores a rater is likely to give to a candidate with a given true ability.
        Wider, flatter curves mean more rating error — the same candidate could receive very different scores from different raters.
        Use the slider to set the true ability level, then compare your ICC (solid curve) against the reference levels.
        The colored dashed curves show how much narrower the spread would be at higher agreement levels.
      </div>

      {/* Slider */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.75rem" }}>
        <div style={{ fontSize: "0.8rem", color: "#aaa", whiteSpace: "nowrap" }}>True score:</div>
        <input
          type="range" min={scaleMin} max={scaleMax} step={0.1} value={trueScore}
          onChange={e => setTrueScore(parseFloat(e.target.value))}
          style={{ flex: 1, accentColor: actualColor }}
        />
        <div style={{ fontFamily: "'DM Mono', monospace", fontSize: "1rem", fontWeight: 700,
                      color: actualColor, minWidth: "2.5rem", textAlign: "right" }}>
          {trueScore.toFixed(1)}
        </div>
      </div>

      {/* Toggle buttons for comparison curves */}
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginBottom: "0.85rem" }}>
        {/* Actual always shown, not toggleable */}
        <div style={{
          padding: "0.2rem 0.6rem", borderRadius: "5px", fontSize: "0.7rem", fontWeight: 700,
          background: `${actualColor}22`, border: `1.5px solid ${actualColor}`,
          color: actualColor, whiteSpace: "nowrap",
        }}>
          ● Your ICC
        </div>
        {comparisons.map(c => {
          const on = shown[c.key];
          return (
            <div key={c.key} onClick={() => toggle(c.key)} style={{
              padding: "0.2rem 0.6rem", borderRadius: "5px", fontSize: "0.7rem", fontWeight: 600,
              cursor: "pointer", whiteSpace: "nowrap", transition: "all 0.15s",
              background: on ? `${c.color}18` : "transparent",
              border: `1.5px solid ${on ? c.color : "#333"}`,
              color: on ? c.color : "#444",
            }}>
              {on ? "●" : "○"} {c.label} ({c.icc})
            </div>
          );
        })}
      </div>

      {/* SVG */}
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible" }}>
        <defs>
          {curvePoints.map(c => (
            <linearGradient key={c.key} id={`grad-${c.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={c.color}
                    stopOpacity={c.key === "actual" ? "0.4" : "0.12"} />
              <stop offset="100%" stopColor={c.color} stopOpacity="0" />
            </linearGradient>
          ))}
        </defs>

        {/* Comparison curves (draw first so actual is on top) */}
        {curvePoints.filter(c => c.key !== "actual" && shown[c.key]).map(c => (
          <g key={c.key}>
            <path d={makeFill(c.pts)} fill={`url(#grad-${c.key})`} />
            <path d={makePath(c.pts)} fill="none" stroke={c.color}
                  strokeWidth={c.strokeW} strokeDasharray={c.dash} opacity="0.7" strokeLinejoin="round" />
          </g>
        ))}

        {/* Actual curve (on top) */}
        {shown.actual !== false && (() => {
          const c = curvePoints.find(c => c.key === "actual");
          return (
            <g>
              <path d={makeFill(c.pts)} fill={`url(#grad-actual)`} />
              <path d={makePath(c.pts)} fill="none" stroke={c.color}
                    strokeWidth={c.strokeW} strokeLinejoin="round" />
            </g>
          );
        })()}

        {/* True score needle */}
        <line x1={toX(trueScore)} y1={PAD_T - 4} x2={toX(trueScore)} y2={PAD_T + plotH}
              stroke="#fff" strokeWidth="1.5" strokeDasharray="3,3" opacity="0.5" />

        {/* X axis */}
        <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH}
              stroke="#2a2d3e" strokeWidth="1" />

        {tickScores.map(t => (
          <g key={t}>
            <line x1={toX(t)} y1={PAD_T + plotH} x2={toX(t)} y2={PAD_T + plotH + 4}
                  stroke="#444" strokeWidth="1" />
            <text x={toX(t)} y={PAD_T + plotH + 14} textAnchor="middle"
                  fill="#555" fontSize="8" fontFamily="DM Mono, monospace">{t}</text>
          </g>
        ))}

        {/* Score category dots */}
        {[
          { x: toX(1), color: "#e74c3c" },
          { x: toX(2), color: "#e67e22" },
          { x: toX(3), color: "#2980b9" },
          { x: toX(4), color: "#27ae60" },
        ].map(({ x, color }, i) => (
          <circle key={i} cx={x} cy={PAD_T + plotH + 22} r="4" fill={color} opacity="0.8" />
        ))}
      </svg>

      {/* Stats row */}
      <div style={{ display: "flex", gap: "1rem", marginTop: "0.4rem",
                    fontSize: "0.72rem", color: "#555", flexWrap: "wrap" }}>
        {curvePoints.filter(c => c.key === "actual" || shown[c.key]).map(c => (
          <span key={c.key}>
            <span style={{ color: c.color, fontWeight: 700 }}>
              {c.key === "actual" ? `Your ICC` : c.label}
            </span>
            {" "}±<span style={{ fontFamily: "DM Mono, monospace", color: c.color }}>
              {c.sd.toFixed(2)}
            </span> SD
          </span>
        ))}
      </div>

      <div style={{ marginTop: "0.6rem", fontSize: "0.72rem", color: "#444", lineHeight: 1.5 }}>
        Solid curve = your ICC. Dashed curves = what spread would look like at benchmark reliability levels.
        Narrower = less noise. Drag the slider to explore boundary cases.
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [step, setStep] = useState("upload"); // upload | options | result
  const [rawText, setRawText] = useState("");
  const [matrix, setMatrix] = useState(null);
  const [headers, setHeaders] = useState(null);
  const [agreementType, setAgreementType] = useState("absolute");
  const [cutPoint, setCutPoint] = useState(3);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState("");

  const processFile = useCallback((text, name) => {
    setError("");
    try {
      const { matrix, headers } = parseCSV(text);
      if (matrix.length < 2) throw new Error("Need at least 2 rows (cases).");
      if (matrix[0].length < 2) throw new Error("Need at least 2 columns (raters).");
      setMatrix(matrix);
      setHeaders(headers);
      setRawText(text);
      setFileName(name || "data");
      setStep("options");
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => processFile(e.target.result, file.name);
    reader.readAsText(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleCalculate = () => {
    const res = calculateICC(matrix, agreementType);
    setResult(res);
    setStep("result");
  };

  const reset = () => {
    setStep("upload"); setMatrix(null); setHeaders(null);
    setResult(null); setError(""); setRawText(""); setFileName("");
  };

  const loadSimulated = (targetICC, label) => {
    const mat = generateSimulatedData(targetICC);
    setMatrix(mat);
    setHeaders(["Rater1", "Rater2", "Rater3"]);
    setFileName(`${label} (simulated)`);
    setRawText("");
    setError("");
    setCutPoint(3);
    setStep("options");
  };

  // ── Styles ────────────────────────────────────────────────────────────────
  const s = {
    app: {
      minHeight: "100vh",
      background: "#0f1117",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "2rem",
      fontFamily: "'DM Sans', sans-serif",
    },
    card: {
      background: "#1a1d27",
      borderRadius: "20px",
      padding: "2.5rem",
      width: "100%",
      maxWidth: "640px",
      boxShadow: "0 24px 64px rgba(0,0,0,0.5)",
      border: "1px solid #2a2d3e",
    },
    header: {
      marginBottom: "2rem",
    },
    eyebrow: {
      fontSize: "0.7rem",
      fontWeight: 700,
      letterSpacing: "0.15em",
      color: "#f5a623",
      textTransform: "uppercase",
      marginBottom: "0.4rem",
    },
    title: {
      fontSize: "1.6rem",
      fontWeight: 700,
      color: "#f0f0f0",
      lineHeight: 1.2,
      margin: 0,
    },
    subtitle: {
      fontSize: "0.85rem",
      color: "#666",
      marginTop: "0.4rem",
    },
    dropzone: {
      border: `2px dashed ${dragging ? "#f5a623" : "#2a2d3e"}`,
      borderRadius: "12px",
      padding: "2.5rem 1.5rem",
      textAlign: "center",
      cursor: "pointer",
      transition: "all 0.2s",
      background: dragging ? "rgba(245,166,35,0.05)" : "#141720",
    },
    dropIcon: {
      fontSize: "2.5rem",
      marginBottom: "0.75rem",
    },
    dropText: {
      color: "#888",
      fontSize: "0.9rem",
    },
    dropHint: {
      color: "#555",
      fontSize: "0.75rem",
      marginTop: "0.4rem",
    },
    fileInput: {
      display: "none",
    },
    btn: {
      display: "inline-block",
      padding: "0.5rem 1.2rem",
      borderRadius: "8px",
      border: "1px solid #f5a623",
      background: "transparent",
      color: "#f5a623",
      cursor: "pointer",
      fontSize: "0.85rem",
      fontWeight: 600,
      marginTop: "1rem",
      fontFamily: "'DM Sans', sans-serif",
      transition: "all 0.15s",
    },
    btnPrimary: {
      background: "#f5a623",
      color: "#0f1117",
      border: "none",
      padding: "0.75rem 2rem",
      borderRadius: "10px",
      fontSize: "1rem",
      fontWeight: 700,
      cursor: "pointer",
      width: "100%",
      marginTop: "1.5rem",
      fontFamily: "'DM Sans', sans-serif",
      boxShadow: "0 4px 16px rgba(245,166,35,0.3)",
      transition: "all 0.15s",
    },
    label: {
      fontSize: "0.75rem",
      fontWeight: 700,
      letterSpacing: "0.08em",
      color: "#888",
      textTransform: "uppercase",
      marginBottom: "0.75rem",
      display: "block",
    },
    optionCard: (active) => ({
      border: `2px solid ${active ? "#f5a623" : "#2a2d3e"}`,
      borderRadius: "12px",
      padding: "1.2rem 1.5rem",
      cursor: "pointer",
      background: active ? "rgba(245,166,35,0.06)" : "#141720",
      marginBottom: "0.75rem",
      transition: "all 0.15s",
    }),
    optionTitle: (active) => ({
      fontSize: "0.95rem",
      fontWeight: 700,
      color: active ? "#f5a623" : "#ccc",
      marginBottom: "0.2rem",
    }),
    optionDesc: {
      fontSize: "0.8rem",
      color: "#666",
      lineHeight: 1.5,
    },
    statGrid: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr 1fr",
      gap: "0.75rem",
      marginBottom: "1.5rem",
    },
    statBox: {
      background: "#141720",
      borderRadius: "10px",
      padding: "0.9rem",
      textAlign: "center",
      border: "1px solid #2a2d3e",
    },
    statVal: {
      fontSize: "1.4rem",
      fontWeight: 700,
      color: "#f0f0f0",
      fontFamily: "'DM Mono', monospace",
    },
    statLbl: {
      fontSize: "0.7rem",
      color: "#666",
      marginTop: "0.2rem",
      textTransform: "uppercase",
      letterSpacing: "0.08em",
    },
    divider: {
      borderColor: "#2a2d3e",
      margin: "1.5rem 0",
    },
    errorBox: {
      background: "rgba(231,76,60,0.1)",
      border: "1px solid rgba(231,76,60,0.3)",
      borderRadius: "8px",
      padding: "0.75rem 1rem",
      color: "#e74c3c",
      fontSize: "0.85rem",
      marginTop: "1rem",
    },
    backBtn: {
      background: "none",
      border: "none",
      color: "#666",
      cursor: "pointer",
      fontSize: "0.8rem",
      padding: 0,
      marginBottom: "1.5rem",
      display: "flex",
      alignItems: "center",
      gap: "0.3rem",
      fontFamily: "'DM Sans', sans-serif",
    },
    gaugeWrap: {
      background: "#141720",
      borderRadius: "14px",
      padding: "1.5rem",
      border: "1px solid #2a2d3e",
      marginBottom: "1.5rem",
    },
    interpretNote: {
      fontSize: "0.78rem",
      color: "#555",
      lineHeight: 1.6,
      marginTop: "1rem",
    },
    tag: {
      display: "inline-block",
      background: "rgba(245,166,35,0.12)",
      color: "#f5a623",
      borderRadius: "6px",
      padding: "0.15rem 0.5rem",
      fontSize: "0.75rem",
      fontWeight: 600,
      marginRight: "0.4rem",
    },
  };

  const [reportLoading, setReportLoading] = useState(false);

  // Load docx library from CDN dynamically
  const loadDocx = () => new Promise((resolve, reject) => {
    if (window.docx) return resolve(window.docx);
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/docx/8.5.0/docx.umd.min.js";
    script.onload = () => resolve(window.docx);
    script.onerror = reject;
    document.head.appendChild(script);
  });

  const getZoneLabel = (v) => {
    if (v < 0) return "Negative";
    if (v < 0.50) return "Poor";
    if (v < 0.60) return "Borderline";
    if (v < 0.75) return "Moderate";
    if (v < 0.90) return "Good";
    if (v < 0.975) return "Excellent";
    return "Suspiciously Ideal";
  };

  const generateReport = async () => {
    setReportLoading(true);
    try {
      // 1. Get AI narrative from Anthropic API
      const iccLabel = getZoneLabel(result.icc);
      const scaleVariance = (3 * 3) / 12;
      const errorSD = Math.sqrt((1 - Math.max(0, Math.min(0.9999, result.icc))) * scaleVariance);

      // Misclassification stats (1,000 applicants, uniform true scores)
      const thresh = cutPoint - 0.5;
      const pAtCut  = normalCDF((thresh - cutPoint) / errorSD);
      const pAbove  = normalCDF((thresh - (cutPoint + 1)) / errorSD);
      const falseFromCut  = Math.round(250 * pAtCut);
      const falseFromAbove = Math.round(250 * pAbove);
      const totalFalse = falseFromCut + falseFromAbove;
      const falsePct = ((totalFalse / 500) * 100).toFixed(1);

      const prompt = `You are writing a brief section of a professional inter-rater reliability report.
Write 2–3 concise paragraphs interpreting the following ICC results for a non-technical audience (e.g. a hiring manager).

Data:
- ICC value: ${result.icc.toFixed(3)}
- Agreement type: ${agreementType === "absolute" ? "Absolute Agreement ICC(2,1)" : "Consistency ICC(2,1)"}
- Reliability category: ${iccLabel}
- 95% Confidence Interval: [${result.ciLow.toFixed(3)}, ${result.ciHigh.toFixed(3)}]
- Number of cases rated: ${result.n}
- Number of raters: ${result.k}
- Rating scale: 1–4, passing cut point: ${cutPoint}
- Estimated error SD: ±${errorSD.toFixed(2)} score points
- Assuming 1,000 applicants with uniformly distributed true scores, scores < ${cutPoint} are rejected:
  - Estimated false rejections (truly score ${cutPoint}, rated < ${cutPoint}): ${falseFromCut} applicants
  - Estimated false rejections (truly score ${cutPoint + 1}, rated < ${cutPoint}): ${falseFromAbove} applicants
  - Total incorrectly rejected: ${totalFalse} of 500 truly-qualified applicants (${falsePct}%)

Cover: what the ICC score means in plain language, what the practical implication is for using these ratings to make decisions, including specific mention of the false rejection numbers, and one concrete recommendation. Be direct and specific. Do not use jargon. Do not use bullet points.`;

      let narrative = "";
      try {
        const resp = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-20250514",
            max_tokens: 1000,
            messages: [{ role: "user", content: prompt }],
          }),
        });
        const data = await resp.json();
        narrative = data.content?.find(b => b.type === "text")?.text || "";
      } catch {
        narrative = `The ICC of ${result.icc.toFixed(3)} indicates ${iccLabel.toLowerCase()} inter-rater reliability across ${result.n} cases and ${result.k} raters. Please interpret this result with reference to the interpretation guide below.`;
      }

      // 2. Build docx
      const docx = await loadDocx();
      const {
        Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, BorderStyle, WidthType, ShadingType, HeadingLevel,
        VerticalAlign,
      } = docx;

      const DARK = "3D4F6B";
      const ORANGE = "F5A623";
      const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
      const borders = { top: border, bottom: border, left: border, right: border };
      const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

      const makeCell = (text, opts = {}) => new TableCell({
        borders,
        width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
        shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
        verticalAlign: VerticalAlign.CENTER,
        margins: cellMargins,
        children: [new Paragraph({
          alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
          children: [new TextRun({
            text,
            bold: opts.bold || false,
            color: opts.color || "222222",
            size: opts.size || 20,
            font: "Arial",
          })]
        })]
      });

      // Zone rows for interpretation table
      const zoneRows = [
        ["< 0.00",      "Negative",           "C0392B", "Raters agree less than chance. Serious rubric or training problem."],
        ["0.00 – 0.50", "Poor",                "E74C3C", "Agreement insufficient. Consider rater training or rubric revision."],
        ["0.50 – 0.60", "Borderline",          "F39C12", "Marginal agreement. Proceed with caution; review outlier raters."],
        ["0.60 – 0.75", "Moderate",            "D4AC0D", "Acceptable for exploratory use. Not recommended for high-stakes decisions alone."],
        ["0.75 – 0.90", "Good",                "27AE60", "Good reliability. Suitable for most selection contexts."],
        ["0.90 – 0.975","Excellent",           "1E8449", "Excellent agreement. Strong confidence in rater consistency."],
        ["> 0.975",     "Suspiciously Ideal",  "7F8C8D", "Unusually high. Check for rater collusion or data entry issues."],
      ];

      const doc = new Document({
        styles: {
          default: { document: { run: { font: "Arial", size: 20 } } },
          paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 28, bold: true, font: "Arial", color: DARK },
              paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 22, bold: true, font: "Arial", color: DARK },
              paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
          ]
        },
        sections: [{
          properties: {
            page: { size: { width: 12240, height: 15840 },
                    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
          },
          children: [
            // Title
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { before: 0, after: 80 },
              border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ORANGE } },
              shading: { fill: DARK, type: ShadingType.CLEAR },
              children: [new TextRun({ text: "Inter-Rater Reliability Report", bold: true,
                                       size: 36, color: "FFFFFF", font: "Arial" })]
            }),
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { before: 0, after: 320 },
              shading: { fill: DARK, type: ShadingType.CLEAR },
              children: [new TextRun({ text: `Generated ${new Date().toLocaleDateString("en-US", { year:"numeric", month:"long", day:"numeric" })}`,
                                       size: 18, color: "AAAAAA", font: "Arial" })]
            }),

            // Summary stats table
            new Paragraph({ heading: HeadingLevel.HEADING_1,
              children: [new TextRun({ text: "Summary Statistics", font: "Arial" })] }),
            new Table({
              width: { size: 9360, type: WidthType.DXA },
              columnWidths: [2340, 2340, 2340, 2340],
              rows: [
                new TableRow({ children: [
                  makeCell("ICC Value", { fill: DARK, bold: true, color: "FFFFFF", width: 2340, center: true }),
                  makeCell("Category",  { fill: DARK, bold: true, color: "FFFFFF", width: 2340, center: true }),
                  makeCell("Cases",     { fill: DARK, bold: true, color: "FFFFFF", width: 2340, center: true }),
                  makeCell("Raters",    { fill: DARK, bold: true, color: "FFFFFF", width: 2340, center: true }),
                ]}),
                new TableRow({ children: [
                  makeCell(result.icc.toFixed(3), { fill: "EEF2F7", bold: true, width: 2340, center: true, size: 24 }),
                  makeCell(iccLabel,               { fill: "EEF2F7", width: 2340, center: true }),
                  makeCell(String(result.n),       { fill: "EEF2F7", width: 2340, center: true }),
                  makeCell(String(result.k),       { fill: "EEF2F7", width: 2340, center: true }),
                ]}),
                new TableRow({ children: [
                  makeCell(`Agreement type: ${agreementType === "absolute" ? "Absolute ICC(2,1)" : "Consistency ICC(2,1)"}`,
                    { fill: "F5F5F5", width: 2340, size: 18 }),
                  makeCell(`95% CI: [${result.ciLow.toFixed(3)}, ${result.ciHigh.toFixed(3)}]`,
                    { fill: "F5F5F5", width: 2340, size: 18 }),
                  makeCell(`Error SD: ±${errorSD.toFixed(2)} pts`, { fill: "F5F5F5", width: 2340, size: 18 }),
                  makeCell(`Scale: 1–4`, { fill: "F5F5F5", width: 2340, size: 18 }),
                ]}),
              ]
            }),

            // Narrative
            new Paragraph({ spacing: { before: 320, after: 0 }, children: [new TextRun("")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1,
              children: [new TextRun({ text: "Interpretation", font: "Arial" })] }),
            ...narrative.split("\n").filter(p => p.trim()).map(para =>
              new Paragraph({
                spacing: { before: 0, after: 160 },
                children: [new TextRun({ text: para.trim(), size: 20, font: "Arial", color: "333333" })]
              })
            ),

            // Interpretation guide table
            new Paragraph({ spacing: { before: 200, after: 0 }, children: [new TextRun("")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1,
              children: [new TextRun({ text: "Reliability Scale Reference", font: "Arial" })] }),
            new Table({
              width: { size: 9360, type: WidthType.DXA },
              columnWidths: [1400, 1600, 6360],
              rows: [
                new TableRow({ children: [
                  makeCell("Range",    { fill: DARK, bold: true, color: "FFFFFF", width: 1400, center: true }),
                  makeCell("Category", { fill: DARK, bold: true, color: "FFFFFF", width: 1600, center: true }),
                  makeCell("What it means", { fill: DARK, bold: true, color: "FFFFFF", width: 6360 }),
                ]}),
                ...zoneRows.map(([range, label, color, desc], i) =>
                  new TableRow({ children: [
                    makeCell(range, { fill: i % 2 === 0 ? "F5F5F5" : "FFFFFF", width: 1400, center: true, size: 18 }),
                    makeCell(label, { fill: color, bold: true, color: "FFFFFF", width: 1600, center: true, size: 18 }),
                    makeCell(desc,  { fill: i % 2 === 0 ? "F5F5F5" : "FFFFFF", width: 6360, size: 18 }),
                  ]})
                ),
              ]
            }),

            // Misclassification section
            new Paragraph({ spacing: { before: 200, after: 0 }, children: [new TextRun("")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1,
              children: [new TextRun({ text: "False Rejection Estimate (1,000 Applicants)", font: "Arial" })] }),
            new Paragraph({
              spacing: { before: 0, after: 120 },
              children: [new TextRun({
                text: `Assuming 1,000 applicants with uniformly distributed true scores (250 per score point) and a pass threshold of score ≥ 3, the following estimates how many truly-qualified candidates would be incorrectly rejected by a single rater under this level of agreement.`,
                size: 20, font: "Arial", color: "333333",
              })]
            }),
            new Table({
              width: { size: 9360, type: WidthType.DXA },
              columnWidths: [3120, 3120, 3120],
              rows: [
                new TableRow({ children: [
                  makeCell("Group",                    { fill: DARK, bold: true, color: "FFFFFF", width: 3120, center: true }),
                  makeCell("False Rejections (of 250)", { fill: DARK, bold: true, color: "FFFFFF", width: 3120, center: true }),
                  makeCell("Error Rate",               { fill: DARK, bold: true, color: "FFFFFF", width: 3120, center: true }),
                ]}),
                new TableRow({ children: [
                  makeCell("True score 3 → rated ≤ 2", { fill: "FFF8EE", width: 3120 }),
                  makeCell(String(falseFrom3),           { fill: "FFF8EE", width: 3120, center: true, bold: true, size: 22 }),
                  makeCell(`${(p3 * 100).toFixed(1)}%`,  { fill: "FFF8EE", width: 3120, center: true }),
                ]}),
                new TableRow({ children: [
                  makeCell("True score 4 → rated ≤ 2", { fill: "F5F5F5", width: 3120 }),
                  makeCell(String(falseFrom4),           { fill: "F5F5F5", width: 3120, center: true, bold: true, size: 22 }),
                  makeCell(`${(p4 * 100).toFixed(1)}%`,  { fill: "F5F5F5", width: 3120, center: true }),
                ]}),
                new TableRow({ children: [
                  makeCell("Total incorrectly rejected", { fill: "EEF2F7", width: 3120, bold: true }),
                  makeCell(`${totalFalse} of 500`,        { fill: "EEF2F7", width: 3120, center: true, bold: true, size: 24 }),
                  makeCell(`${falsePct}%`,                { fill: "EEF2F7", width: 3120, center: true, bold: true }),
                ]}),
              ]
            }),
            new Paragraph({
              spacing: { before: 80, after: 0 },
              children: [new TextRun({
                text: "Assumes scores ≤ 2 are rejected. Error model: single-rater observation with ICC-implied measurement noise (normal distribution). True score distribution assumed uniform across 1–4.",
                size: 16, italics: true, color: "888888", font: "Arial",
              })]
            }),

            // Footer note
            new Paragraph({ spacing: { before: 400, after: 0 },
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: ORANGE } },
              children: [new TextRun({ text: "Report generated by the LEEP ICC Rater Agreement Calculator. ICC computed using the two-way random effects model (Shrout & Fleiss, 1979).",
                                       size: 16, italics: true, color: "888888", font: "Arial" })]
            }),
          ]
        }]
      });

      const buffer = await Packer.toBuffer(doc);
      const blob = new Blob([buffer], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `ICC_Report_${new Date().toISOString().slice(0,10)}.docx`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
    } finally {
      setReportLoading(false);
    }
  };
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
      `}</style>
      <div style={s.app}>
        <div style={s.card}>

          {/* ── UPLOAD ── */}
          {step === "upload" && (
            <>
              <div style={s.header}>
                <div style={s.eyebrow}>Rater Reliability</div>
                <h1 style={s.title}>ICC Agreement Calculator</h1>
                <p style={s.subtitle}>Upload ratings data to measure inter-rater reliability using Intraclass Correlation.</p>
              </div>

              <label>
                <input
                  type="file"
                  accept=".csv,.tsv,.txt"
                  style={s.fileInput}
                  onChange={e => handleFile(e.target.files[0])}
                />
                <div
                  style={s.dropzone}
                  onDragOver={e => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                >
                  <div style={s.dropIcon}>📊</div>
                  <div style={s.dropText}>Drop your CSV or TSV file here</div>
                  <div style={s.dropHint}>Rows = cases &nbsp;·&nbsp; Columns = raters &nbsp;·&nbsp; Cells = scores</div>
                  <div style={{
                    display: "inline-block",
                    marginTop: "1rem",
                    padding: "0.4rem 1rem",
                    borderRadius: "6px",
                    border: "1px solid #f5a623",
                    color: "#f5a623",
                    fontSize: "0.8rem",
                    fontWeight: 600,
                  }}>Browse file</div>
                </div>
              </label>

              <div style={{ marginTop: "1.5rem" }}>
                <div style={s.label}>Or paste data directly</div>
                <textarea
                  placeholder={"Rater1,Rater2,Rater3\n3,4,3\n2,2,3\n4,4,4\n1,2,1"}
                  value={rawText}
                  onChange={e => setRawText(e.target.value)}
                  style={{
                    width: "100%",
                    minHeight: "100px",
                    background: "#141720",
                    border: "1px solid #2a2d3e",
                    borderRadius: "10px",
                    color: "#ccc",
                    fontFamily: "'DM Mono', monospace",
                    fontSize: "0.8rem",
                    padding: "0.75rem",
                    resize: "vertical",
                    outline: "none",
                  }}
                />
                <button
                  style={s.btnPrimary}
                  onClick={() => processFile(rawText, "pasted data")}
                >
                  Load Data →
                </button>
              </div>

              {error && <div style={s.errorBox}>⚠ {error}</div>}

              {/* Sample data */}
              <div style={{ marginTop: "1.75rem" }}>
                <div style={s.label}>Or try with simulated data</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.6rem" }}>
                  {[
                    { label: "High Agreement", icc: 0.90, color: "#27ae60", desc: "ICC ≈ 0.90" },
                    { label: "Moderate",        icc: 0.70, color: "#f1c40f", desc: "ICC ≈ 0.70" },
                    { label: "Low Agreement",   icc: 0.35, color: "#e74c3c", desc: "ICC ≈ 0.35" },
                  ].map(({ label, icc, color, desc }) => (
                    <button
                      key={label}
                      onClick={() => loadSimulated(icc, label)}
                      style={{
                        background: `${color}12`,
                        border: `1.5px solid ${color}55`,
                        borderRadius: "10px",
                        padding: "0.75rem 0.5rem",
                        cursor: "pointer",
                        color,
                        fontFamily: "'DM Sans', sans-serif",
                        textAlign: "center",
                        transition: "all 0.15s",
                      }}
                    >
                      <div style={{ fontWeight: 700, fontSize: "0.8rem" }}>{label}</div>
                      <div style={{ fontSize: "0.7rem", opacity: 0.75, marginTop: "0.2rem" }}>{desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* ── OPTIONS ── */}
          {step === "options" && (
            <>
              <button style={s.backBtn} onClick={reset}>← Back</button>

              <div style={s.header}>
                <div style={s.eyebrow}>Data Loaded</div>
                <h1 style={s.title}>Configure Analysis</h1>
              </div>

              <div style={s.statGrid}>
                <div style={s.statBox}>
                  <div style={s.statVal}>{matrix?.length}</div>
                  <div style={s.statLbl}>Cases</div>
                </div>
                <div style={s.statBox}>
                  <div style={s.statVal}>{matrix?.[0]?.length}</div>
                  <div style={s.statLbl}>Raters</div>
                </div>
                <div style={s.statBox}>
                  <div style={s.statVal}>{fileName.split(".")[0].slice(0, 8)}</div>
                  <div style={s.statLbl}>Source</div>
                </div>
              </div>

              {headers && (
                <div style={{ marginBottom: "1.5rem" }}>
                  <div style={s.label}>Rater columns detected</div>
                  <div>{headers.map(h => <span key={h} style={s.tag}>{h}</span>)}</div>
                </div>
              )}

              <hr style={s.divider} />

              <div style={s.label}>Agreement type</div>

              <div
                style={s.optionCard(agreementType === "absolute")}
                onClick={() => setAgreementType("absolute")}
              >
                <div style={s.optionTitle(agreementType === "absolute")}>
                  {agreementType === "absolute" ? "● " : "○ "}Absolute Agreement
                </div>
                <div style={s.optionDesc}>
                  Raters must give the <strong style={{ color: "#aaa" }}>same scores</strong> to agree.
                  Use this when exact numeric scores matter — e.g. comparing raters who may be systematically
                  higher or lower than each other. Reports <strong style={{ color: "#aaa" }}>ICC(2,1) Absolute</strong>.
                </div>
              </div>

              <div
                style={s.optionCard(agreementType === "relative")}
                onClick={() => setAgreementType("relative")}
              >
                <div style={s.optionTitle(agreementType === "relative")}>
                  {agreementType === "relative" ? "● " : "○ "}Relative Ordering (Consistency)
                </div>
                <div style={s.optionDesc}>
                  Raters agree if they <strong style={{ color: "#aaa" }}>rank cases the same way</strong>,
                  even if one rater scores systematically higher. Use this when the ordering of candidates
                  matters more than the exact scores. Reports <strong style={{ color: "#aaa" }}>ICC(2,1) Consistency</strong>.
                </div>
              </div>

              <hr style={s.divider} />

              <div style={s.label}>Passing score cut point</div>
              <div style={{ fontSize: "0.72rem", color: "#555", marginBottom: "0.75rem" }}>
                Cases scored at or above this value are considered passing. Used in false rejection and per-rater analysis.
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                {[1, 1.5, 2, 2.5, 3, 3.5, 4].map(v => (
                  <button
                    key={v}
                    onClick={() => setCutPoint(v)}
                    style={{
                      flex: 1, padding: "0.5rem 0", borderRadius: "8px", cursor: "pointer",
                      fontSize: "0.8rem", fontWeight: 700, fontFamily: "'DM Mono', monospace",
                      border: cutPoint === v ? "2px solid #4f8ef7" : "1px solid #2a2d3e",
                      background: cutPoint === v ? "rgba(79,142,247,0.15)" : "#1a1d27",
                      color: cutPoint === v ? "#4f8ef7" : "#555",
                      transition: "all 0.15s",
                    }}
                  >{v}</button>
                ))}
              </div>
              <div style={{ fontSize: "0.68rem", color: "#444", marginTop: "0.5rem" }}>
                Selected: score ≥ {cutPoint} = pass, score &lt; {cutPoint} = fail
              </div>

              <button style={s.btnPrimary} onClick={handleCalculate}>
                Calculate ICC →
              </button>
            </>
          )}

          {/* ── RESULT ── */}
          {step === "result" && result && (
            <>
              <button style={s.backBtn} onClick={() => setStep("options")}>← Adjust settings</button>

              <div style={s.header}>
                <div style={s.eyebrow}>
                  {agreementType === "absolute" ? "ICC(2,1) Absolute Agreement" : "ICC(2,1) Consistency"}
                </div>
                <h1 style={s.title}>Rater Agreement Results</h1>
              </div>

              <div style={s.statGrid}>
                <div style={s.statBox}>
                  <div style={s.statVal}>{result.n}</div>
                  <div style={s.statLbl}>Cases</div>
                </div>
                <div style={s.statBox}>
                  <div style={s.statVal}>{result.k}</div>
                  <div style={s.statLbl}>Raters</div>
                </div>
                <div style={s.statBox}>
                  <div style={s.statVal}>{agreementType === "absolute" ? "Abs." : "Con."}</div>
                  <div style={s.statLbl}>Type</div>
                </div>
              </div>

              <div style={s.gaugeWrap}>
                <ICCGauge
                  icc={result.icc}
                  ciLow={result.ciLow}
                  ciHigh={result.ciHigh}
                />
              </div>

              <div style={{
                background: "#141720",
                borderRadius: "10px",
                padding: "1rem 1.25rem",
                border: "1px solid #2a2d3e",
                marginBottom: "1rem",
              }}>
                <div style={{ fontSize: "0.75rem", color: "#666", textTransform: "uppercase",
                              letterSpacing: "0.08em", marginBottom: "0.5rem", fontWeight: 700 }}>
                  Interpretation Guide
                </div>
                {[
                  ["> 0.975",      "Suspiciously Ideal", "#95a5a6", "Unusually high. Check for rater collusion or data entry issues."],
                  ["0.90 – 0.975", "Excellent",          "#1e90ff", "Excellent agreement. Strong confidence in rater consistency."],
                  ["0.75 – 0.90",  "Good",               "#27ae60", "Good reliability. Suitable for most selection contexts."],
                  ["0.60 – 0.75",  "Moderate",           "#f1c40f", "Acceptable for exploratory use. Not recommended for high-stakes decisions alone."],
                  ["0.50 – 0.60",  "Borderline",         "#e74c3c", "Marginal agreement. Proceed with caution; review outlier raters."],
                  ["0.00 – 0.50",  "Poor",               "#95a5a6", "Rater agreement is insufficient. Consider rater training or rubric revision."],
                ].map(([range, label, color, desc]) => (
                  <div key={range} style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start",
                                            marginBottom: "0.5rem", fontSize: "0.78rem" }}>
                    <span style={{ color, fontFamily: "'DM Mono', monospace", minWidth: "80px",
                                   fontWeight: 600, flexShrink: 0 }}>{range}</span>
                    <span style={{ color: "#aaa" }}><strong style={{ color }}>{label}</strong> — {desc}</span>
                  </div>
                ))}
              </div>

              <NoiseVisualization
                icc={result.icc}
                scaleMin={1}
                scaleMax={4}
              />

              <div style={{ display: "flex", gap: "0.75rem", marginBottom: "0.75rem" }}>
                <button
                  style={{ ...s.btnPrimary, marginTop: 0, flex: 1, background: "#2a2d3e",
                           color: "#ccc", boxShadow: "none" }}
                  onClick={() => setStep("options")}
                >
                  ← Change Type
                </button>
                <button
                  style={{ ...s.btnPrimary, marginTop: 0, flex: 1 }}
                  onClick={reset}
                >
                  New Analysis
                </button>
              </div>

              <MisclassificationPanel icc={result.icc} cutPoint={cutPoint} />
              <PerRaterBreakdown matrix={matrix} headers={headers} cutPoint={cutPoint} />
            </>
          )}

          {/* ── CHAT WIDGET (all steps) ── */}
          <ChatWidget
            iccContext={result ? {
              icc: result.icc.toFixed(3),
              ciLow: result.ciLow.toFixed(3),
              ciHigh: result.ciHigh.toFixed(3),
              n: result.n,
              k: result.k,
              label: (() => {
                const v = result.icc;
                if (v < 0) return "Negative";
                if (v < 0.50) return "Poor";
                if (v < 0.60) return "Borderline";
                if (v < 0.75) return "Moderate";
                if (v < 0.90) return "Good";
                if (v < 0.975) return "Excellent";
                return "Suspiciously Ideal";
              })(),
              agreementType,
              cutPoint,
            } : null}
          />

        </div>
      </div>
    </>
  );
}
