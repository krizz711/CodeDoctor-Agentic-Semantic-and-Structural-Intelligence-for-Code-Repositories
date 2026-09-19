/**
 * Build the Samsung PRISM Theme 01 submission deck from docs/SUBMISSION-DECK.md.
 *
 * The markdown is the source of truth for the words and the numbers; this file
 * is the layout. Keeping them separate means the deck can be regenerated after
 * a measurement changes without anyone re-typing a figure into a text box.
 *
 *   npm install pptxgenjs
 *   node docs/deck/build_deck.js
 *
 * Writes docs/CodeDoctor-PRISM-Theme01.pptx
 */

const path = require("path");
const PptxGenJS = require("pptxgenjs");

// --- palette -------------------------------------------------------------
// The deck's argument is which passes cleared a significance gate and which
// did not, so the accent system is literally pass/reject and gets used that
// way on every slide that reports a measurement.
const INK = "111A2E"; // dark slide background
const INK_SOFT = "1C2949"; // raised panel on dark
const PAPER = "FFFFFF";
const SLATE = "22304F"; // headings on light
const BODY = "39435A";
const MUTED = "7C879B";
const TINT = "EEF2F8"; // card fill on light
const VERIFIED = "0F9B82"; // cleared the gate
const REJECTED = "C65B3F"; // did not clear
const ICE = "BFD2EE";

const SANS = "Calibri";
const MONO = "Courier New";

const W = 13.3;
const M = 0.62; // page margin
const CW = W - 2 * M; // content width

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Munal Singh";
pptx.company = "CodeDoctor";
pptx.title = "CodeDoctor - Agentic Code Intelligence";

// --- helpers -------------------------------------------------------------

function lightSlide(titleText, kicker) {
  const s = pptx.addSlide();
  s.background = { color: PAPER };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.36, w: CW, h: 0.26,
      fontFace: SANS, fontSize: 11, bold: true, color: VERIFIED,
      charSpacing: 1.6, isTextBox: true, margin: 0,
    });
  }
  s.addText(titleText, {
    x: M, y: kicker ? 0.62 : 0.5, w: CW, h: 0.8,
    fontFace: SANS, fontSize: 34, bold: true, color: SLATE,
    isTextBox: true, margin: 0,
  });
  return s;
}

function darkSlide(titleText, kicker) {
  const s = pptx.addSlide();
  s.background = { color: INK };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.36, w: CW, h: 0.26,
      fontFace: SANS, fontSize: 11, bold: true, color: VERIFIED,
      charSpacing: 1.6, isTextBox: true, margin: 0,
    });
  }
  s.addText(titleText, {
    x: M, y: kicker ? 0.62 : 0.5, w: CW, h: 0.8,
    fontFace: SANS, fontSize: 34, bold: true, color: PAPER,
    isTextBox: true, margin: 0,
  });
  return s;
}

/** The recurring motif: a verdict pill on anything that was measured. */
function verdict(s, x, y, text, passed, w) {
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w: w || 1.85, h: 0.32,
    fill: { color: passed ? VERIFIED : REJECTED },
    rectRadius: 0.16, line: { type: "none" },
  });
  s.addText(text.toUpperCase(), {
    x, y, w: w || 1.85, h: 0.32,
    fontFace: SANS, fontSize: 10, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", charSpacing: 1, isTextBox: true, margin: 0,
  });
}

function card(s, x, y, w, h, opts) {
  const o = opts || {};
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: o.fill || TINT },
    rectRadius: 0.07,
    line: o.line || { type: "none" },
  });
}

function body(s, text, x, y, w, h, opts) {
  const o = opts || {};
  s.addText(text, {
    x, y, w, h,
    fontFace: o.mono ? MONO : SANS,
    fontSize: o.size || 14,
    color: o.color || BODY,
    bold: !!o.bold,
    italic: !!o.italic,
    lineSpacingMultiple: o.lineSpacingMultiple || 1.12,
    align: o.align || "left",
    valign: o.valign || "top",
    isTextBox: true,
    margin: o.margin === undefined ? 0 : o.margin,
  });
}

function bullets(s, items, x, y, w, h, opts) {
  const o = opts || {};
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i !== items.length - 1 },
    })),
    {
      x, y, w, h,
      fontFace: SANS, fontSize: o.size || 14, color: o.color || BODY,
      paraSpaceAfter: o.spaceAfter === undefined ? 8 : o.spaceAfter,
      lineSpacingMultiple: 1.1,
      isTextBox: true, margin: 0,
    }
  );
}

/** Big number + caption, the deck's way of stating a measurement. */
function stat(s, x, y, value, caption, color, size) {
  s.addText(value, {
    x, y, w: 3.4, h: 0.95,
    fontFace: SANS, fontSize: size || 44, bold: true, color,
    isTextBox: true, margin: 0,
  });
  s.addText(caption, {
    x, y: y + (size && size > 50 ? 1.0 : 0.86), w: 3.4, h: 0.5,
    fontFace: SANS, fontSize: 11, color: MUTED,
    lineSpacingMultiple: 1.05, isTextBox: true, margin: 0,
  });
}

function table(s, rows, x, y, w, opts) {
  const o = opts || {};
  const dark = !!o.dark;
  s.addTable(rows, {
    x, y, w,
    colW: o.colW,
    fontFace: SANS,
    fontSize: o.size || 12,
    color: dark ? "D8E0EE" : BODY,
    border: { type: "solid", pt: 0.5, color: dark ? "2E3E63" : "D7DEE9" },
    fill: { color: dark ? INK_SOFT : "FFFFFF" },
    align: "left",
    valign: "middle",
    rowH: o.rowH || 0.3,
    margin: 0.07,
  });
}

function hdr(text) {
  return {
    text,
    options: { bold: true, color: "FFFFFF", fill: { color: SLATE }, fontSize: 11.5 },
  };
}

function hdrDark(text) {
  return {
    text,
    options: { bold: true, color: INK, fill: { color: ICE }, fontSize: 11.5 },
  };
}

function footer(s, n, dark) {
  s.addText(`CodeDoctor  ·  Samsung PRISM GenAI Hackathon 2026  ·  Theme 01`, {
    x: M, y: 6.92, w: CW - 0.6, h: 0.3,
    fontFace: SANS, fontSize: 9, color: dark ? "5E6C88" : MUTED,
    isTextBox: true, margin: 0,
  });
  s.addText(String(n), {
    x: W - M - 0.6, y: 6.92, w: 0.6, h: 0.3,
    fontFace: SANS, fontSize: 9, color: dark ? "5E6C88" : MUTED,
    align: "right", isTextBox: true, margin: 0,
  });
}

// =========================================================================
// 1 — Title
// =========================================================================
{
  const s = pptx.addSlide();
  s.background = { color: INK };

  s.addText("CodeDoctor", {
    x: M, y: 1.15, w: 8.6, h: 1.1,
    fontFace: SANS, fontSize: 58, bold: true, color: PAPER,
    isTextBox: true, margin: 0,
  });
  s.addText("Semantic and structural intelligence for code repositories", {
    x: M, y: 2.22, w: 8.8, h: 0.5,
    fontFace: SANS, fontSize: 19, color: ICE, isTextBox: true, margin: 0,
  });

  card(s, M, 3.35, 5.3, 1.85, { fill: INK_SOFT });
  s.addText("0.96558", {
    x: M + 0.35, y: 3.5, w: 4.6, h: 0.95,
    fontFace: SANS, fontSize: 54, bold: true, color: VERIFIED,
    isTextBox: true, margin: 0,
  });
  s.addText("NDCG@10  ·  CoIR AppsRetrieval test split  ·  MTEB 2.21.0", {
    x: M + 0.37, y: 4.48, w: 4.9, h: 0.5,
    fontFace: SANS, fontSize: 11, color: ICE, isTextBox: true, margin: 0,
  });

  s.addText("0.93756", {
    x: M + 5.85, y: 3.62, w: 3.0, h: 0.6,
    fontFace: SANS, fontSize: 30, bold: true, color: "8FA2C4",
    isTextBox: true, margin: 0,
  });
  s.addText("frozen encoder alone\n(published: 0.93690)", {
    x: M + 5.87, y: 4.24, w: 3.2, h: 0.6,
    fontFace: SANS, fontSize: 11, color: MUTED,
    lineSpacingMultiple: 1.1, isTextBox: true, margin: 0,
  });

  s.addText("Three passes over one frozen CPU encoder — and an honest account of\nwhat each pass bought, including the two that bought nothing.", {
    x: M, y: 5.55, w: 9.4, h: 0.7,
    fontFace: SANS, fontSize: 13, color: "A9B7D0",
    lineSpacingMultiple: 1.2, isTextBox: true, margin: 0,
  });

  s.addText("Samsung PRISM GenAI Hackathon 2026\nTheme 01 — Agentic Code Intelligence", {
    x: M, y: 6.55, w: 8, h: 0.6,
    fontFace: SANS, fontSize: 11, bold: true, color: MUTED,
    lineSpacingMultiple: 1.15, isTextBox: true, margin: 0,
  });
  s.addNotes(
    "Headline: 0.96558 NDCG@10 on the CoIR AppsRetrieval test split, against " +
    "0.93756 for the frozen encoder alone. The encoder reproduces its published " +
    "score, so everything above that line is this project's contribution."
  );
}

// =========================================================================
// 2 — Problem
// =========================================================================
{
  const s = lightSlide("Rank code by what a sentence means", "The problem");
  body(s,
    "Given a library of code and a natural-language query, rank the snippets by relevance.",
    M, 1.62, CW, 0.4, { size: 17, color: SLATE });

  const items = [
    ["Thousands", "of snippets in the corpus"],
    ["Arbitrary", "snippet length — 225 tokens on average, 289,002 at the tail"],
    ["CPU only", "no GPU at query time"],
    ["Sub-second", "retrieval is the first step, so it cannot be the slow one"],
  ];
  const cw = (CW - 3 * 0.26) / 4;
  items.forEach((it, i) => {
    const x = M + i * (cw + 0.26);
    card(s, x, 2.35, cw, 1.65);
    body(s, it[0], x + 0.24, 2.58, cw - 0.48, 0.42, { size: 19, bold: true, color: VERIFIED });
    body(s, it[1], x + 0.24, 3.04, cw - 0.48, 0.85, { size: 12 });
  });

  card(s, M, 4.35, CW, 0.85, { fill: "F7F3EF" });
  body(s, "Not in scope", M + 0.3, 4.53, 2.0, 0.3, { size: 12, bold: true, color: REJECTED });
  body(s, "Generating an answer.  Explaining the results.  This is the retrieval stage only.",
    M + 2.1, 4.53, CW - 2.4, 0.4, { size: 13 });

  body(s,
    "The corpus is competitive-programming Python: problem statements as queries, the programs that solve them as documents. " +
    "A statement says “find the last two digits of 5^n”; the answer is a loop over int(input()). " +
    "Query and document share almost no vocabulary, which is why there is no lexical pass in this system.",
    M, 5.5, CW, 1.1, { size: 13, color: MUTED });
  footer(s, 2);
  s.addNotes("Lexical retrieval is close to useless on this corpus — that is a measured property, not an assumption, and it is why there is no BM25 pass.");
}

// =========================================================================
// 3 — Architecture
// =========================================================================
{
  const s = lightSlide("One frozen encoder, two passes on top", "Architecture");

  const boxW = 4.1, boxH = 0.74;
  const lx = M + 0.5, rx = W - M - 0.5 - boxW;

  card(s, lx, 1.62, boxW, boxH, { fill: TINT });
  body(s, "Categorize the query", lx + 0.2, 1.75, boxW - 0.4, 0.28, { size: 13, bold: true, color: SLATE });
  body(s, "family · identifiers · literals", lx + 0.2, 2.04, boxW - 0.4, 0.26, { size: 11, color: MUTED });

  card(s, rx, 1.62, boxW, boxH, { fill: TINT });
  body(s, "Categorize the corpus", rx + 0.2, 1.75, boxW - 0.4, 0.28, { size: 13, bold: true, color: SLATE });
  body(s, "AST facts · answer shape", rx + 0.2, 2.04, boxW - 0.4, 0.26, { size: 11, color: MUTED });

  const wideX = M + 1.55, wideW = CW - 3.1;

  card(s, wideX, 2.72, wideW, 0.82, { fill: SLATE });
  body(s, "Frozen dense encoder — F2LLM-v2, CPU, published recipe pinned by test",
    wideX + 0.2, 2.93, wideW - 0.4, 0.4, { size: 14, bold: true, color: "FFFFFF", align: "center" });

  body(s, "top-50 candidate pool", wideX, 3.6, wideW, 0.26,
    { size: 10.5, color: MUTED, align: "center" });

  card(s, wideX, 3.92, wideW, 0.82, { fill: TINT });
  body(s, "Pass 2 — nine-feature linear reranker", wideX + 0.25, 4.14, wideW - 2.6, 0.4,
    { size: 14, bold: true, color: SLATE });
  verdict(s, wideX + wideW - 2.1, 4.17, "did not clear", false);

  card(s, wideX, 5.02, wideW, 0.82, { fill: TINT });
  body(s, "Pass 3 — execution verification", wideX + 0.25, 5.24, wideW - 2.6, 0.4,
    { size: 14, bold: true, color: SLATE });
  verdict(s, wideX + wideW - 2.1, 5.27, "cleared the gate", true);

  body(s, "ranked snippets", wideX, 5.98, wideW, 0.3,
    { size: 13, bold: true, color: VERIFIED, align: "center" });

  body(s, "Every addition is measured against the pass below it on a held-out split, and only ships if the gain survives a paired bootstrap.",
    M, 6.42, CW, 0.4, { size: 12.5, italic: true, color: MUTED });
  footer(s, 3);
}

// =========================================================================
// 4 — The recipe bug
// =========================================================================
{
  const s = lightSlide("2.8 points were sitting on the floor", "Reproduction first");

  stat(s, M, 1.6, "0.87603", "our earlier build", REJECTED, 40);
  s.addText("→", {
    x: M + 2.1, y: 1.66, w: 0.7, h: 0.8,
    fontFace: SANS, fontSize: 32, color: MUTED, align: "center", isTextBox: true, margin: 0,
  });
  stat(s, M + 2.85, 1.6, "0.90446", "the same model's published score", VERIFIED, 40);

  body(s, "Same model. Same data. The gap was four configuration values.",
    M, 2.95, 6.6, 0.35, { size: 13.5, bold: true, color: SLATE });

  table(s, [
    [hdr(""), hdr("earlier build"), hdr("published recipe")],
    ["query instruction", "generic", "Retrieve the most relevant code snippet…"],
    ["document prefix", "—", "none (correct)"],
    ["add_eos_token", "not set", "true"],
    ["max_seq_length", "512 doc / 1024 query", "8192"],
  ], M, 3.42, 7.35, { colW: [1.95, 2.0, 3.4], rowH: 0.32, size: 11 });

  card(s, M + 7.7, 1.6, CW - 7.7, 3.35, { fill: "FBF1ED" });
  body(s, "Why it is a wrong vector, not a worse one", M + 7.95, 1.82, CW - 8.2, 0.6,
    { size: 14, bold: true, color: REJECTED });
  body(s,
    "F2LLM pools with last-token pooling. Truncation cuts the EOS, so the pooled vector becomes an arbitrary interior token.\n\n" +
    "4.58% of documents and 1.73% of queries were being silently corrupted — and the run still succeeded every time.",
    M + 7.95, 2.5, CW - 8.2, 2.2, { size: 12.5, lineSpacingMultiple: 1.18 });

  card(s, M, 5.32, CW, 1.0, { fill: TINT });
  body(s, "tests/test_published_recipe.py pins all four values against MTEB's own model registry, and the encoder probes the loaded tokenizer at construction and refuses to run if the EOS is missing. This cannot regress silently.",
    M + 0.3, 5.5, CW - 0.6, 0.7, { size: 12.5, color: SLATE });

  body(s, "On this benchmark, reproducing the recipe exactly was worth more than most algorithmic ideas we tried afterwards.",
    M, 6.45, CW, 0.35, { size: 12.5, italic: true, color: MUTED });
  footer(s, 4);
  s.addNotes("This is the single highest-value change in the project, and it is not an algorithm. Worth saying plainly.");
}

// =========================================================================
// 5 — Pre/post processing
// =========================================================================
{
  const s = lightSlide("A query implies a shape of answer", "Brief axes 1–4");

  table(s, [
    [hdr("query family"), hdr("surface evidence"), hdr("implied answer")],
    ["competitive-stdin", "-----Input-----, $n$, “the first line contains”", "reads input(), prints"],
    ["leetcode-method", "Example 1:, Constraints:, nums", "class Solution:"],
    ["kata-function", "markdown fences, “write a function”", "a free-standing def"],
    [
      { text: "unknown", options: { bold: true, color: REJECTED } },
      { text: "—", options: {} },
      { text: "no prior is applied", options: { bold: true, color: REJECTED } },
    ],
  ], M, 1.68, CW, { colW: [2.5, 5.6, 3.98], rowH: 0.38, size: 12 });

  card(s, M, 4.0, 6.0, 1.15, { fill: TINT });
  body(s, "UNKNOWN is a real outcome", M + 0.28, 4.18, 5.4, 0.3, { size: 13.5, bold: true, color: SLATE });
  body(s, "A family is asserted only on evidence that clears an absolute floor and leads the runner-up. Otherwise the query contributes nothing downstream — rather than a guess.",
    M + 0.28, 4.5, 5.45, 0.6, { size: 12 });

  card(s, M + 6.28, 4.0, CW - 6.28, 1.15, { fill: TINT });
  body(s, "The fallback labels itself", M + 6.56, 4.18, 5.4, 0.3, { size: 13.5, bold: true, color: SLATE });
  body(s, "ast.parse per snippet, with a regex fallback for the Python-2 and truncated fragments in the corpus. The fallback reports CANDIDATE, never VERIFIED.",
    M + 6.56, 4.5, CW - 6.84, 0.6, { size: 12 });

  body(s, "Both sides of the pair are categorized before they are compared, and both are allowed to say they do not know.",
    M, 5.45, CW, 0.4, { size: 13, italic: true, color: MUTED });
  footer(s, 5);
}

// =========================================================================
// 6 — The second pass
// =========================================================================
{
  const s = lightSlide("Nine features, linear, and printable", "Brief axis 5");

  const feats = ["dense", "dense_margin", "family_compat", "defined_name_hit", "ident_idf_overlap",
    "number_idf_overlap", "string_idf_overlap", "length_prior", "hub_penalty"];
  let fx = M, fy = 1.62;
  feats.forEach((f) => {
    const w = 0.13 * f.length + 0.5;
    if (fx + w > M + CW) { fx = M; fy += 0.48; }
    s.addShape(pptx.ShapeType.roundRect, {
      x: fx, y: fy, w, h: 0.38, fill: { color: TINT }, rectRadius: 0.19, line: { type: "none" },
    });
    s.addText(f, {
      x: fx, y: fy, w, h: 0.38, fontFace: MONO, fontSize: 11, color: SLATE,
      align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    fx += w + 0.16;
  });

  const rows = [
    ["Linear, on purpose", "One relevant document per query and nine features. A linear score is as accurate as anything heavier here, runs in microseconds, and lets the demo print why a snippet moved."],
    ["Weights capped at ±2.0", "Cosine gaps inside a top-50 pool are 0.01–0.1. Unbounded, coordinate ascent ran one weight to −6.31 — at which point the encoder stops mattering and the system is a hand-built classifier the dev split cannot vouch for."],
    ["IDF is corpus-side only", "Adding a query changes nothing, so a weight measured on one split transfers to another. Inductive, not transductive."],
  ];
  let ry = 2.72;
  rows.forEach((r, i) => {
    s.addShape(pptx.ShapeType.ellipse, {
      x: M, y: ry + 0.03, w: 0.42, h: 0.42, fill: { color: SLATE }, line: { type: "none" },
    });
    s.addText(String(i + 1), {
      x: M, y: ry + 0.03, w: 0.42, h: 0.42, fontFace: SANS, fontSize: 13, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    body(s, r[0], M + 0.62, ry + 0.05, 3.1, 0.3, { size: 13.5, bold: true, color: SLATE });
    body(s, r[1], M + 3.85, ry, CW - 3.85, 1.0, { size: 12 });
    ry += 1.1;
  });

  card(s, M, 6.12, CW, 0.74, { fill: TINT });
  body(s, "Fitted by coordinate ascent directly on NDCG@10 — the reported metric, not a surrogate.",
    M + 0.3, 6.32, 8.4, 0.35, { size: 12.5, color: SLATE });
  verdict(s, W - M - 2.35, 6.32, "did not clear", false, 2.05);
  footer(s, 6);
}

// =========================================================================
// 7 — The measurement problem
// =========================================================================
{
  const s = lightSlide("Train and test are different datasets", "Measurement");

  s.addChart(pptx.ChartType.bar, [
    {
      name: "train (5,000)",
      labels: ["competitive-stdin", "kata", "leetcode", "unknown"],
      values: [34.1, 43.0, 14.6, 8.2],
    },
    {
      name: "test (3,765)",
      labels: ["competitive-stdin", "kata", "leetcode", "unknown"],
      values: [98.4, 0.1, 0.8, 0.8],
    },
  ], {
    x: M, y: 1.66, w: 6.75, h: 3.5,
    barDir: "col",
    chartColors: ["9FB2CE", VERIFIED],
    showTitle: true,
    title: "Query-family mix, from query text alone (%)",
    titleFontFace: SANS, titleFontSize: 12, titleColor: SLATE,
    showValue: true,
    dataLabelPosition: "outEnd",
    // Without an explicit format these round to integers, which turns the
    // whole point of the slide -- 0.1% vs 43.0% -- into "0" against "43".
    dataLabelFormatCode: "0.0",
    dataLabelFontFace: SANS, dataLabelFontSize: 9, dataLabelColor: BODY,
    catAxisLabelFontFace: SANS, catAxisLabelFontSize: 9.5, catAxisLabelColor: BODY,
    valAxisLabelFontFace: SANS, valAxisLabelFontSize: 9, valAxisLabelColor: MUTED,
    valAxisMinVal: 0,
    valAxisMaxVal: 110,
    valGridLine: { color: "E6EBF2", size: 1 },
    catGridLine: { style: "none" },
    showLegend: true,
    legendPos: "t",
    legendFontFace: SANS, legendFontSize: 10, legendColor: BODY,
  });

  body(s, "The corpus is ordered by source, and the test split is a contiguous tail of it.",
    M + 7.05, 1.72, CW - 7.05, 0.5, { size: 13, bold: true, color: SLATE });

  card(s, M + 7.05, 2.35, CW - 7.05, 1.32, { fill: TINT });
  body(s, "Consequence 1", M + 7.3, 2.52, 4.6, 0.28, { size: 12.5, bold: true, color: REJECTED });
  body(s, "A contiguous half-split of train is not exchangeable — the halves score 0.209 and 0.623. A “held-out” number taken that way reports distribution shift, not generalization.",
    M + 7.3, 2.82, CW - 7.55, 0.8, { size: 11.5 });

  card(s, M + 7.05, 3.82, CW - 7.05, 1.32, { fill: TINT });
  body(s, "Consequence 2", M + 7.3, 3.99, 4.6, 0.28, { size: 12.5, bold: true, color: REJECTED });
  body(s, "A uniform sample of train is the wrong dev set too. We tune on the 1,707 competitive-stdin train queries, split at random and seeded.",
    M + 7.3, 4.29, CW - 7.55, 0.8, { size: 11.5 });

  body(s, "Everyone on this leaderboard is validating against a distribution that is not the one being scored. It is the most consequential thing we measured, and it is free to check.",
    M, 5.45, CW, 0.5, { size: 13, italic: true, color: MUTED });
  footer(s, 7);
  s.addNotes("This finding reshaped the whole evaluation setup. Without it, every dev number we produced would have been measuring the wrong thing.");
}

// =========================================================================
// 8 — What we refused to use
// =========================================================================
{
  const s = lightSlide("Three shortcuts we did not take", "Integrity");

  const items = [
    ["meta_information", "Carries the original problem URL and, for kata problems, the starter code — which names the exact function signature of the answer."],
    ["the partition column", "Marks each document train or test. Using it would halve the candidate set for free."],
    ["the qrels pattern", "The gold map is the identity q<N> → d<N>. Nothing stops a system from simply reading it."],
  ];
  const cw = (CW - 2 * 0.3) / 3;
  items.forEach((it, i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 1.68, cw, 2.5);
    s.addShape(pptx.ShapeType.roundRect, {
      x: x + 0.24, y: 1.92, w: 0.44, h: 0.44,
      fill: { color: REJECTED }, rectRadius: 0.1, line: { type: "none" },
    });
    s.addText("✕", {
      x: x + 0.24, y: 1.92, w: 0.44, h: 0.44, fontFace: SANS, fontSize: 15, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    body(s, it[0], x + 0.24, 2.52, cw - 0.48, 0.34, { size: 14, bold: true, color: SLATE, mono: false });
    body(s, it[1], x + 0.24, 2.92, cw - 0.48, 1.15, { size: 12 });
  });

  card(s, M, 4.42, CW, 1.15, { fill: TINT });
  body(s, "The structure enforces it, not a promise", M + 0.3, 4.6, 6.0, 0.3, { size: 13.5, bold: true, color: SLATE });
  body(s, "MTEB hands the encoder text and nothing else. The pipeline records the texts it was given during encode() and reads nothing else in similarity(). Each of these three would raise the reported score without improving retrieval at all.",
    M + 0.3, 4.92, CW - 0.6, 0.6, { size: 12.5 });

  body(s, "We know the shortcuts exist. The code is shaped so it cannot reach them.",
    M, 5.85, CW, 0.4, { size: 13, italic: true, color: MUTED });
  footer(s, 8);
}

// =========================================================================
// 9 — Execution verification
// =========================================================================
{
  const s = lightSlide("Evidence no embedding can supply", "Pass 3");
  verdict(s, W - M - 2.3, 0.68, "cleared the gate", true, 2.3);

  body(s, "Competitive-programming statements publish worked examples. Feed a statement's sample input to a candidate program and compare its stdout.",
    M, 1.62, 8.4, 0.5, { size: 14, color: SLATE });

  stat(s, M, 2.32, "97.6%", "of test statements yield a parseable\nworked example  (train: 32.0%)", VERIFIED, 38);
  stat(s, M + 3.5, 2.32, "71–82%", "of gold solutions reproduce\ntheir own sample output", SLATE, 38);
  stat(s, M + 7.0, 2.32, "0 / 100", "random non-gold candidates\npassed a statement's examples", SLATE, 38);

  card(s, M, 4.02, 7.0, 1.6, { fill: "FBF1ED" });
  body(s, "It executes dataset-authored code", M + 0.28, 4.22, 6.4, 0.3, { size: 13.5, bold: true, color: REJECTED });
  body(s, "Fresh python -I -S subprocess · throwaway working directory · stripped environment · 2-second wall-clock timeout · address-space cap where the platform allows. Never in-process. Every failure mode returns False rather than raising.",
    M + 0.28, 4.56, 6.45, 0.95, { size: 12 });

  card(s, M + 7.28, 4.02, CW - 7.28, 1.6, { fill: TINT });
  body(s, "Promotion only, never demotion", M + 7.56, 4.22, 4.5, 0.3, { size: 13.5, bold: true, color: SLATE });
  body(s, "A verified candidate is promoted. A failed one is never demoted: a crash may be an environment difference rather than a wrong answer, so evidence promotes and absence proves nothing.",
    M + 7.56, 4.56, CW - 7.84, 0.95, { size: 12 });

  card(s, M, 5.85, CW, 0.85, { fill: TINT });
  body(s, "Shipped on — and defended, not hidden. It is the only pass that cleared the paired-bootstrap gate, and leaving it on is what makes a bare reproduce command match our submitted number.  --no-execution reports the dense-only score.",
    M + 0.3, 6.06, CW - 0.6, 0.55, { size: 12.5, color: SLATE });
  footer(s, 9);
}

// =========================================================================
// 10 — Results (hero, dark)
// =========================================================================
{
  const s = darkSlide("What each pass was actually worth", "Results");

  body(s, "Held-out DEV — 853 competitive-stdin queries the weights never saw. Every delta from a 1000-resample paired bootstrap.",
    M, 1.56, CW, 0.32, { size: 12.5, color: ICE });

  table(s, [
    [hdrDark("stage"), hdrDark("NDCG@10"), hdrDark("MRR@10"), hdrDark("R@10"), hdrDark("Δ vs dense (95% CI)")],
    ["dense only", "0.89659", "0.87560", "0.9613", "—"],
    [
      { text: "+ feature reranker", options: {} },
      { text: "0.90139", options: {} },
      { text: "0.87997", options: {} },
      { text: "0.9613", options: {} },
      { text: "+0.0048  [−0.0009, +0.0109]   rejected", options: { color: "E79D86" } },
    ],
    ["+ execution, top-10", "0.94061", "0.93382", "0.9613", "+0.0440  [+0.0342, +0.0532]"],
    [
      { text: "+ execution, top-20", options: { bold: true, color: "FFFFFF" } },
      { text: "0.95486", options: { bold: true, color: "FFFFFF" } },
      { text: "0.94730", options: { bold: true, color: "FFFFFF" } },
      { text: "0.9777", options: { bold: true, color: "FFFFFF" } },
      { text: "+0.0583  [+0.0457, +0.0702]", options: { bold: true, color: "5FD9C2" } },
    ],
  ], M, 1.98, CW, { dark: true, colW: [2.5, 1.5, 1.5, 1.3, 5.26], rowH: 0.32, size: 11 });

  body(s, "Test split — 3,765 queries, scored by mteb.evaluate 2.21.0.",
    M, 3.92, 7.6, 0.3, { size: 12.5, color: ICE });

  table(s, [
    [hdrDark("test-split stage"), hdrDark("NDCG@10"), hdrDark("MRR@10"), hdrDark("R@10"), hdrDark("sandbox")],
    ["dense only, F2LLM-v2-1.7B", "0.93756", "0.92370", "0.97955", "—"],
    ["+ execution, top-10", "0.96085", "0.95449", "0.97955", "5,839 s"],
    ["+ top-20, exact compare", "0.96543", "0.95837", "0.98619", "10,686 s"],
    [
      { text: "+ numeric compare (shipped)", options: { bold: true, color: "FFFFFF" } },
      { text: "0.96558", options: { bold: true, color: "5FD9C2" } },
      { text: "0.95857", options: { bold: true, color: "FFFFFF" } },
      { text: "0.98619", options: { bold: true, color: "FFFFFF" } },
      { text: "11,920 s", options: {} },
    ],
  ], M, 4.28, 8.55, { dark: true, colW: [3.0, 1.45, 1.45, 1.35, 1.3], rowH: 0.30, size: 10.5 });

  card(s, M + 8.85, 3.92, CW - 8.85, 1.94, { fill: INK_SOFT });
  body(s, "Say the negative result out loud", M + 9.1, 4.1, 3.3, 0.3, { size: 12.5, bold: true, color: "E79D86" });
  body(s, "Our own nine-feature reranker did not clear its own gate. Against a weak encoder it is worth +0.037; against F2LLM-v2-1.7B, +0.0048 with a CI spanning zero. The tuner shipped the identity config and recorded why. Fusion (+0.0038, p = 0.164) and PRF were rejected the same way.",
    M + 9.1, 4.42, CW - 9.35, 1.35, { size: 10.5, color: "C3CEE2" });

  body(s, "The last step is not claimed as a gain.", M, 6.02, 4.2, 0.3, { size: 12.5, bold: true, color: "FFFFFF" });
  body(s, "Numeric comparison is +0.00015 over exact — 95% CI [−0.00046, +0.00081], p = 0.630, four queries better and three worse. It ships as the higher measured value under a more correct rule, not because the score moved. On held-out DEV the same change was +0.00468 with an interval excluding zero; it did not transfer, because after execution the test split has almost no headroom left. Depth 20 over depth 10 was the real step: 25 golds recovered from ranks 11–20 against 13 lost to false verification, +8 rank-1 hits.",
    M, 6.3, CW, 0.62, { size: 10.5, color: "A9B7D0" });
  footer(s, 10, true);
  s.addNotes("The rejected reranker is on this slide deliberately. A gate that never rejects anything is not a gate.");
}

// =========================================================================
// 11 — Speed
// =========================================================================
{
  const s = lightSlide("Under a second, on a deliberately bad laptop", "Speed");
  body(s, "Measured end to end on a 6-core i5-11260H with 7.7 GB RAM, and reported as measured.",
    M, 1.6, CW, 0.32, { size: 13, color: SLATE });

  table(s, [
    [hdr("operation"), hdr("cost"), hdr("note")],
    ["query classification", "1.2 ms", "pure regex"],
    ["query encode (0.6B, CPU)", "924 ms", "dominates; this box is RAM-starved"],
    ["dense search over 8,765 snippets", "21 ms", "one numpy matmul"],
    ["feature rerank of top 50", "0 ms", "identity config — it did not ship"],
    [
      { text: "free-text query, total", options: { bold: true, color: SLATE } },
      { text: "947 ms", options: { bold: true, color: VERIFIED } },
      { text: "no worked examples, so no execution", options: {} },
    ],
    ["execution verification (top 10)", "+361 ms", "only fires on statements with examples"],
    [
      { text: "dataset query, total", options: { bold: true, color: SLATE } },
      { text: "2,077 ms", options: { bold: true, color: VERIFIED } },
      { text: "", options: {} },
    ],
    ["corpus encode, 17,530 texts", "12 / 24 min", "one-time, on one T4 (0.6B / 1.7B)"],
  ], M, 2.05, CW, { colW: [4.3, 2.0, 5.78], rowH: 0.36, size: 12 });

  card(s, M, 5.55, 6.0, 1.15, { fill: TINT });
  body(s, "The index is built once", M + 0.28, 5.73, 5.4, 0.3, { size: 13.5, bold: true, color: SLATE });
  body(s, "Only the query is encoded at query time — which is why a query answers in under a second against 8,765 snippets on a laptop. It is the same code path the service runs.",
    M + 0.28, 6.05, 5.45, 0.6, { size: 12 });

  card(s, M + 6.28, 5.55, CW - 6.28, 1.15, { fill: TINT });
  body(s, "Two tiers, one recipe", M + 6.56, 5.73, 5.4, 0.3, { size: 13.5, bold: true, color: SLATE });
  body(s, "1.7B for the screening submission, 0.6B for interactive use. A tier changes which model runs, never how it is prompted.",
    M + 6.56, 6.05, CW - 6.84, 0.6, { size: 12 });
  footer(s, 11);
}

// =========================================================================
// 12 — Versions and evolution
// =========================================================================
{
  const s = lightSlide("Repositories change; the index keeps up", "P1 and the bonus goal");

  const items = [
    ["Zero re-embeddings on a rename", "The embedding cache is keyed on H(encoder_rev, tokenizer_rev, template_rev, encoder_input_bytes). Path and line numbers are deliberately excluded, so moving or renaming a file triggers no re-embedding at all."],
    ["~0.01 s to re-index one edited function", "Incremental at the symbol level, not the file level."],
    ["0 leaks across 50 randomized property trials", "Commit-pinned queries restrict candidates to that commit's tree snapshot."],
    ["100.0% span fidelity", "489 chunks across 171 files round-trip exactly — a chunk's recorded span always reproduces its source text."],
  ];
  let y = 1.68;
  items.forEach((it) => {
    card(s, M, y, CW, 1.12);
    s.addShape(pptx.ShapeType.ellipse, {
      x: M + 0.3, y: y + 0.35, w: 0.42, h: 0.42, fill: { color: VERIFIED }, line: { type: "none" },
    });
    s.addText("✓", {
      x: M + 0.3, y: y + 0.35, w: 0.42, h: 0.42, fontFace: SANS, fontSize: 14, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    body(s, it[0], M + 0.94, y + 0.2, 4.6, 0.32, { size: 13.5, bold: true, color: SLATE });
    body(s, it[1], M + 0.94, y + 0.54, CW - 1.3, 0.5, { size: 11.5 });
    y += 1.24;
  });
  footer(s, 12);
}

// =========================================================================
// 13 — Engineering notes
// =========================================================================
{
  const s = darkSlide("Five things that cost us a day each", "Engineering notes");

  const notes = [
    ["Attention memory is quadratic", "Documents run 225 tokens on average and 289,002 at the tail. A linear token budget still OOMed at 12 GiB; batches are budgeted on max_len² × batch."],
    ["T4 has no native bfloat16", "Torch emulates it and the encode slows by roughly 10×. select_dtype picks fp16 below compute capability 8.0."],
    ["The recipe is verified at load, not assumed", "tokenizer_kwargs is deprecated in sentence-transformers 6. The day it stops being honoured, the EOS quietly vanishes and every embedding is wrong while the run still succeeds. The encoder probes the loaded tokenizer and refuses."],
    ["The hub profile is content-keyed", "It is built against one corpus ordering and applied against MTEB's. Index alignment would have shifted every hub score silently onto the wrong document."],
    ["A planted-noise negative control", "In the test suite, to show why the held-out gate exists: a pure-noise feature still buys in-sample NDCG."],
  ];
  let y = 1.56;
  notes.forEach((n) => {
    s.addShape(pptx.ShapeType.roundRect, {
      x: M, y, w: CW, h: 1.0, fill: { color: INK_SOFT }, rectRadius: 0.06, line: { type: "none" },
    });
    body(s, n[0], M + 0.3, y + 0.14, 4.3, 0.3, { size: 13, bold: true, color: "5FD9C2" });
    body(s, n[1], M + 4.75, y + 0.14, CW - 5.05, 0.75, { size: 11.5, color: "C3CEE2" });
    y += 1.08;
  });
  footer(s, 13, true);
}

const outPath = path.join(__dirname, "..", "CodeDoctor-PRISM-Theme01.pptx");
pptx.writeFile({ fileName: outPath }).then(() => {
  console.log("wrote", outPath);
});
