// numa — interfaccia. I dati arrivano da data/latest.json e data/draws.json,
// rigenerati dall'automazione dopo ogni estrazione (numa.py build).
import * as core from "./core.js";

const REMOTE_DRAWS = "https://raw.githubusercontent.com/Lottopyrhon/Estrazioni_Superenalotto/main/superenalotto.txt";
const ORDER = ["numa", "ensemble", "gbm", "caldi", "ritardatari", "caso"];
const SERIES = { numa: 1, ensemble: 2, gbm: 3, caldi: 4, ritardatari: 5, caso: 6 };
const DESCRIPTIONS = {
  numa: "La meno giocata tra 500 sestine casuali, senza schemi “a disegno”.",
  ensemble: "Il candidato migliore del machine learning, calibrato verso il caso.",
  gbm: "Il gradient boosting che nel 2023–26 sembrava battere il caso. Era un falso positivo.",
  caldi: "I numeri più usciti nelle ultime 50 estrazioni.",
  ritardatari: "I numeri assenti da più tempo: la strategia da bar.",
  caso: "Sei numeri a sorte: il termine di paragone.",
};
const RETRO_NAMES = {
  "caso (uniforme)": "Caso", "freq. storica globale": "Frequenza storica", "numeri caldi (freq. ultime 50)": "Numeri caldi",
  "ritardatari (gap massimo)": "Ritardatari", "regressione logistica": "Regressione logistica",
  "gradient boosting": "Gradient boosting", "LSTM (rete ricorrente)": "Rete neurale LSTM",
  "ensemble temporale calibrato": "Ensemble calibrato",
};

const $ = (s) => document.querySelector(s);
const euro = new Intl.NumberFormat("it-IT", { style: "currency", currency: "EUR" });
const euro0 = new Intl.NumberFormat("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const num = new Intl.NumberFormat("it-IT");
const dec = (x, d = 2) => new Intl.NumberFormat("it-IT", { minimumFractionDigits: d, maximumFractionDigits: d }).format(x);
const pct = (x, d = 1) => `${dec(100 * x, d)}%`;
const signedPct = (x, d = 1) => `${x >= 0 ? "+" : "−"}${dec(Math.abs(100 * x), d)}%`;
// "il 9", "l'11", "l'8": articolo davanti ai numeri che iniziano per vocale
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const the = (n) => ([1, 8, 11].includes(n) || (n >= 80 && n <= 89) ? `l'${n}` : `il ${n}`);
const dateIt = (iso, opts = { day: "numeric", month: "long", year: "numeric" }) =>
  new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString("it-IT", opts);

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) node.append(c instanceof Node ? c : String(c));
  return node;
}

function ball(n, { kind = "", hit = false, small = false, tag = null, muted = false } = {}) {
  const cls = ["ball", kind && `ball--${kind}`, hit && "ball--hit", small && "ball--sm", muted && "ball--muted"].filter(Boolean).join(" ");
  const label = kind === "jolly" ? `Jolly ${n}` : kind === "star" ? `SuperStar ${n}` : `${n}${hit ? ", indovinato" : ""}`;
  return h("span", { class: cls, role: "img", "aria-label": label }, String(n), tag ? h("span", { class: "tag", "aria-hidden": "true" }, tag) : null);
}

// ------------------------------------------------------------ tooltip
const tip = $("#tooltip");
function showTip(evt, ...content) {
  tip.replaceChildren(...content);
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  const x = Math.min(evt.clientX + 14, window.innerWidth - r.width - 8);
  const y = evt.clientY + 16 + r.height > window.innerHeight ? evt.clientY - r.height - 12 : evt.clientY + 16;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; };

// ------------------------------------------------------------ SVG
const charts = [];
function mount(container, draw) {
  charts.push(() => draw(container));
  draw(container);
}
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => charts.forEach((f) => f()), 150); });
const widthOf = (el) => Math.max(300, Math.round(el.clientWidth || 720));
const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  for (const c of children.flat()) if (c) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
// barra con estremo arrotondato (4px) e base squadrata
function barPath(x, y, w, hgt, horizontal = false) {
  const r = Math.min(4, (horizontal ? hgt : w) / 2, horizontal ? w : hgt);
  if (horizontal) return `M${x},${y} h${w - r} q${r},0 ${r},${r} v${hgt - 2 * r} q0,${r} -${r},${r} h-${w - r} z`;
  return `M${x},${y + hgt} v-${hgt - r} q0,-${r} ${r},-${r} h${w - 2 * r} q${r},0 ${r},${r} v${hgt - r} z`;
}
const nice = (max) => { const p = 10 ** Math.floor(Math.log10(max)); return Math.ceil(max / p) * p; };

function hbarChart(container, rows, { ref = null, refLabel = "", format = (v) => num.format(v), highlight = null, min = 0 } = {}) {
  const W = widthOf(container), rowH = 34, left = Math.min(170, Math.round(W * 0.34)), right = 64, top = 22;
  const H = top + rows.length * rowH + 10;
  const max = Math.max(...rows.map((r) => r.value), ref ?? 0) * 1.05;
  const x = (v) => left + ((v - min) / (max - min)) * (W - left - right);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": rows.map((r) => `${r.label}: ${format(r.value)}`).join("; ") });
  rows.forEach((r, i) => {
    const y = top + i * rowH + 7;
    const color = r.key === highlight ? "var(--series-1)" : "var(--axis)";
    svg.append(s("text", { x: left - 10, y: y + 14, "text-anchor": "end", class: r.key === highlight ? "label-strong" : "" }, r.label));
    const bar = s("path", { d: barPath(x(min), y, Math.max(1, x(r.value) - x(min)), 20, true), fill: color, class: "bar" });
    svg.append(bar, s("text", { x: x(r.value) + 6, y: y + 14, class: "label-strong" }, format(r.value)));
    const hit = s("rect", { x: 0, y: y - 6, width: W, height: rowH, class: "hit" });
    hit.addEventListener("pointermove", (e) => { bar.classList.add("is-hover"); showTip(e, h("div", { class: "tv" }, format(r.value)), h("div", {}, r.tip ?? r.label)); });
    hit.addEventListener("pointerleave", () => { bar.classList.remove("is-hover"); hideTip(); });
    svg.append(hit);
  });
  svg.append(s("line", { x1: x(min), x2: x(min), y1: top - 4, y2: H - 6, class: "axis" }));
  if (ref !== null) {
    svg.append(s("line", { x1: x(ref), x2: x(ref), y1: top - 8, y2: H - 6, class: "ref" }));
    svg.append(s("text", { x: x(ref), y: 12, "text-anchor": "middle", class: "halo" }, refLabel));
  }
  container.replaceChildren(svg);
}

function lineChart(container, series, { zeroLabel = "" } = {}) {
  const n = Math.max(...series.map((sr) => sr.values.length));
  if (n < 2) {
    container.replaceChildren(h("p", { class: "empty" }, "Il grafico compare dopo i primi due concorsi verificati: torna dopo la prossima estrazione."));
    return;
  }
  const W = widthOf(container), H = 260, left = 44, right = 16, top = 14, bottom = 28;
  const all = series.flatMap((sr) => sr.values);
  const ext = Math.max(1, ...all.map(Math.abs)) * 1.15;
  const x = (i) => left + (i / (n - 1)) * (W - left - right);
  const y = (v) => top + ((ext - v) / (2 * ext)) * (H - top - bottom);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Punti cumulati rispetto al caso per ogni strategia" });
  const step = nice(ext / 2) / 2 || 1;
  for (let v = -Math.floor(ext / step) * step; v <= ext; v += step) {
    svg.append(s("line", { x1: left, x2: W - right, y1: y(v), y2: y(v), class: Math.abs(v) < 1e-9 ? "axis" : "grid" }));
    svg.append(s("text", { x: left - 8, y: y(v) + 4, "text-anchor": "end" }, (v > 0 ? "+" : "") + dec(v, step < 1 ? 1 : 0)));
  }
  svg.append(s("text", { x: W - right, y: y(0) - 6, "text-anchor": "end" }, zeroLabel));
  for (const sr of series) {
    const d = sr.values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
    svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const last = sr.values.length - 1;
    svg.append(s("circle", { cx: x(last), cy: y(sr.values[last]), r: 4, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2 }));
  }
  const cross = s("line", { y1: top, y2: H - bottom, class: "axis", visibility: "hidden" });
  const hit = s("rect", { x: left, y: top, width: W - left - right, height: H - top - bottom, class: "hit" });
  hit.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - left) / (W - left - right)) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    const rows = series.filter((sr) => i < sr.values.length).sort((a, b) => b.values[i] - a.values[i]).map((sr) =>
      h("div", { class: "row", style: `--c:${sr.color}` }, h("i"), h("strong", {}, (sr.values[i] >= 0 ? "+" : "") + dec(sr.values[i], 1)), " ", sr.label));
    showTip(e, h("div", {}, dateIt(series[0].dates[i] ?? "")), ...rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(cross, hit);
  const legend = h("div", { class: "legend" }, series.map((sr) => h("span", { style: `--c:${sr.color}` }, sr.label)));
  container.replaceChildren(svg, legend);
}

function columnChart(container, values, { expected, label }) {
  const W = widthOf(container), H = 230, left = 40, right = 8, top = 22, bottom = 26;
  const max = nice(Math.max(...values) * 1.05);
  const slot = (W - left - right) / values.length;
  const bw = Math.min(24, slot - 2);
  const y = (v) => top + (1 - v / max) * (H - top - bottom);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label });
  for (let v = 0; v <= max; v += max / 4) {
    svg.append(s("line", { x1: left, x2: W - right, y1: y(v), y2: y(v), class: v === 0 ? "axis" : "grid" }));
    svg.append(s("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end" }, num.format(v)));
  }
  values.forEach((v, i) => {
    const x0 = left + i * slot + (slot - bw) / 2;
    const bar = s("path", { d: barPath(x0, y(v), bw, y(0) - y(v)), fill: "var(--series-1)", class: "bar" });
    const hit = s("rect", { x: left + i * slot, y: top, width: slot, height: H - top - bottom, class: "hit" });
    hit.addEventListener("pointermove", (e) => { bar.classList.add("is-hover"); showTip(e, h("div", { class: "tv" }, `${num.format(v)} volte`), h("div", {}, `${the(i + 1)} · atteso ${num.format(Math.round(expected))}`)); });
    hit.addEventListener("pointerleave", () => { bar.classList.remove("is-hover"); hideTip(); });
    svg.append(bar, hit);
    if ((i + 1) % 10 === 0 || i === 0) svg.append(s("text", { x: x0 + bw / 2, y: H - 8, "text-anchor": "middle" }, i + 1));
  });
  svg.append(s("line", { x1: left, x2: W - right, y1: y(expected), y2: y(expected), class: "ref" }));
  svg.append(s("text", { x: left + 6, y: top - 6, class: "label-strong halo" }, `- - - atteso se tutto è caso: ${num.format(Math.round(expected))}`));
  container.replaceChildren(svg);
}

function dotChart(container, points, { ref }) {
  const W = widthOf(container), H = 230, left = 46, right = 16, top = 16, bottom = 30;
  const max = 0.1;
  const x = (i) => left + ((i + 0.5) / points.length) * (W - left - right);
  const y = (v) => top + (1 - v / max) * (H - top - bottom);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": points.map((p) => `ritardo ${p.gap}+: ${pct(p.p)}`).join("; ") });
  for (const v of [0, 0.025, 0.05, 0.075, 0.1]) {
    svg.append(s("line", { x1: left, x2: W - right, y1: y(v), y2: y(v), class: v === 0 ? "axis" : "grid" }));
    svg.append(s("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end" }, pct(v, 1)));
  }
  svg.append(s("line", { x1: left, x2: W - right, y1: y(ref), y2: y(ref), class: "ref" }));
  svg.append(s("text", { x: W - right, y: y(ref) - 8, "text-anchor": "end", class: "label-strong halo" }, "6,67%: la probabilità di sempre"));
  points.forEach((p, i) => {
    svg.append(s("text", { x: x(i), y: H - 10, "text-anchor": "middle" }, `≥${p.gap}`));
    const half = 1.96 * Math.sqrt((p.p * (1 - p.p)) / p.n);
    svg.append(s("line", { x1: x(i), x2: x(i), y1: y(Math.min(max, p.p + half)), y2: y(Math.max(0, p.p - half)), stroke: "var(--series-2)", "stroke-width": 2, "stroke-linecap": "round", opacity: 0.55 }));
    const dot = s("circle", { cx: x(i), cy: y(p.p), r: 5, fill: "var(--series-2)", stroke: "var(--surface)", "stroke-width": 2 });
    const hit = s("rect", { x: x(i) - 24, y: top, width: 48, height: H - top - bottom, class: "hit" });
    hit.addEventListener("pointermove", (e) => showTip(e, h("div", { class: "tv" }, pct(p.p, 2)),
      h("div", {}, `assente da almeno ${p.gap} concorsi`), h("div", { class: "muted" }, `${num.format(p.n)} casi · margine ±${pct(1.96 * Math.sqrt((p.p * (1 - p.p)) / p.n), 1)}`)));
    hit.addEventListener("pointerleave", hideTip);
    svg.append(dot, hit);
  });
  svg.append(s("text", { x: left, y: H - 10, "text-anchor": "end" }, "g"));
  container.replaceChildren(svg);
}

// ------------------------------------------------------------ dati
let L, D, model, past, lastNums;

async function load() {
  const opts = { cache: "no-cache" };
  [L, D] = await Promise.all([
    fetch("data/latest.json", opts).then((r) => r.json()),
    fetch("data/draws.json", opts).then((r) => r.json()),
  ]);
  model = L.popularity_model;
  past = core.pastIndex(D.rows.map((r) => [dateIt(r[0], { day: "numeric", month: "numeric", year: "numeric" }), r[1]]));
  lastNums = L.last_draw.numbers;
}

// ------------------------------------------------------------ hero
function renderHero() {
  const p = L.prediction;
  const nd = L.next_draw;
  $("#next-label").textContent = `Prossimo concorso · ${nd.label}, ore 20:00`;
  const box = $("#hero-balls");
  if (!p) { box.replaceChildren(h("p", { class: "muted" }, "Le sestine per il prossimo concorso arrivano a breve.")); return; }
  box.replaceChildren(...p.picks.numa.map((n) => ball(n)), p.superstar_hint ? ball(p.superstar_hint, { kind: "star", tag: "★" }) : "");
  const when = new Date(p.created_utc ?? L.generated_utc);
  $("#registered").replaceChildren(p.preview
    ? h("span", {}, "Anteprima locale: questa sestina non è ancora nel registro pubblico.")
    : h("span", {}, `Registrata ${when.toLocaleString("it-IT", { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })}, prima della chiusura delle giocate · impronta `,
      h("code", {}, p.hash.slice(0, 10)), " · ", h("a", { href: "data/ledger.json" }, "registro"),
      p.superstar_hint ? h("span", {}, ` · ★ SuperStar consigliato: ${p.superstar_hint}`) : null));
  tickCountdown();
  setInterval(tickCountdown, 30000);
}

function tickCountdown() {
  const chip = $("#countdown");
  const close = new Date(L.next_draw.sales_close), draw = new Date(L.next_draw.draw_utc), now = new Date();
  chip.hidden = false;
  if (now < close) {
    const m = Math.round((close - now) / 60000), d = Math.floor(m / 1440), hh = Math.floor((m % 1440) / 60), mm = m % 60;
    chip.textContent = `giocate aperte ancora ${d ? `${d} g ` : ""}${hh} h ${mm} min`;
  } else if (now < draw) chip.textContent = "giocate chiuse · estrazione alle 20:00";
  else chip.textContent = "estrazione avvenuta · risultati in arrivo";
}

function generatedRows(tickets) {
  return tickets.map((t) => {
    const rel = core.relativePopularity(t, model);
    return h("div", { class: "row" }, h("div", { class: "balls" }, t.map((n) => ball(n, { small: true }))),
      h("span", { class: "meta" }, `giocata ${dec(rel, 2)}× rispetto a una sestina tipica`));
  });
}

let lastGenerated = [];
function setupGenerator() {
  $("#btn-generate").addEventListener("click", () => {
    const n = Number($("#n-columns").value);
    lastGenerated = core.generate(model, n, { past, last: lastNums });
    $("#generated").replaceChildren(...generatedRows(lastGenerated));
  });
  $("#btn-copy").addEventListener("click", async () => {
    const lines = [L.prediction?.picks.numa, ...lastGenerated].filter(Boolean).map((t) => t.join(" "));
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      $("#btn-copy").textContent = "Copiati ✓";
    } catch (e) {
      $("#btn-copy").textContent = lines[0];
    }
    setTimeout(() => { $("#btn-copy").textContent = "Copia i numeri"; }, 2000);
  });
}

// ------------------------------------------------------------ mascotte
function setupBubble() {
  const theta = model.theta;
  const top = [...theta.keys()].sort((a, b) => theta[b] - theta[a]).slice(0, 3).map((i) => i + 1);
  const gaps = L.stats.gap;
  const late = gaps.indexOf(Math.max(...gaps)) + 1;
  const ld = L.last_draw;
  const tips = [
    "Ciao, sono Numa! Nessuno può prevedere l'estrazione, nemmeno io. Però so quali numeri giocano gli altri.",
    `${cap(the(top[0]))}, ${the(top[1])} e ${the(top[2])} sono tra i numeri più giocati d'Italia. Se escono, si divide con tanta gente.`,
    `${cap(the(late))} manca da ${gaps[late - 1]} concorsi. La probabilità che esca al prossimo? Sempre 6,67%, come tutti.`,
    "Le date di nascita arrivano al 31: troppa gente gioca solo quelle. Io guardo anche più su.",
    ld.pop_3 ? `${dateIt(ld.date, { weekday: "long", day: "numeric", month: "long" })} i numeri usciti erano ${ld.pop_3 < 1 ? "poco" : "molto"} giocati: chi ha fatto 3 ha preso ${euro.format(ld.prizes["3"])}.` : null,
    "Sulla carta i miei numeri valgono come gli altri. Ma quando vinci, incassi in media il 13% in più.",
    "Il vero premio è giocare in compagnia. Gioca poco, e solo quello che puoi permetterti.",
  ].filter(Boolean);
  let i = 0;
  const bubble = $("#bubble");
  bubble.textContent = tips[0];
  const next = () => { i = (i + 1) % tips.length; bubble.textContent = tips[i]; };
  bubble.addEventListener("click", next);
  bubble.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); next(); } });
}

// ------------------------------------------------------------ strategie
function renderStrategies() {
  const p = L.prediction;
  const list = $("#strategy-list");
  if (!p) { list.replaceChildren(h("p", { class: "muted" }, "In arrivo.")); return; }
  const weight = p.info?.ensemble_weight;
  list.replaceChildren(...ORDER.filter((k) => p.picks[k]).map((k) => {
    let desc = DESCRIPTIONS[k];
    if (k === "ensemble" && weight !== undefined) desc += weight === 0 ? " Oggi il peso calibrato è 0: equivale al caso." : ` Peso calibrato: ${dec(weight, 2)}.`;
    return h("div", { class: `strategy${k === "numa" ? " strategy--numa" : ""}` },
      h("div", { class: "name" }, L.labels[k] ?? k, h("span", { class: "desc" }, desc)),
      h("div", { class: "balls" }, p.picks[k].map((n) => ball(n, { small: true }))));
  }));
}

// ------------------------------------------------------------ ultima estrazione
function renderLast() {
  const ld = L.last_draw;
  $("#last-label").textContent = `${ld.label.charAt(0).toUpperCase()}${ld.label.slice(1)}`;
  $("#last-balls").replaceChildren(...ld.numbers.map((n) => ball(n)), ball(ld.jolly, { kind: "jolly", tag: "J" }),
    ld.superstar ? ball(ld.superstar, { kind: "star", tag: "★" }) : "");
  if (ld.pop_3) {
    const r = ld.pop_3;
    const word = r < 0.9 ? "meno giocati della media" : r > 1.1 ? "più giocati della media" : "giocati nella media";
    $("#last-pop").textContent = `Numeri ${word}: c'erano ${dec(r, 2)} vincitori con 3 punti per ogni vincitore atteso se tutti giocassero a caso.`;
  }
  if (ld.prizes) {
    const cats = [["6", "6 punti"], ["5_1", "5+1"], ["5", "5 punti"], ["4", "4 punti"], ["3", "3 punti"], ["2", "2 punti"]];
    $("#last-prizes").replaceChildren(h("caption", {}, "Quota lorda per una colonna vincente"),
      h("tr", {}, h("th", {}, "Categoria"), h("th", { class: "num" }, "Quota")),
      ...cats.filter(([k]) => ld.prizes[k]).map(([k, label]) => h("tr", {}, h("td", {}, label), h("td", { class: "num" },
        k === "6" ? `${euro0.format(ld.prizes[k])}` : euro.format(ld.prizes[k])))));
  }
  const scored = L.recent.filter((e) => e.result);
  $("#recent").replaceChildren(...(scored.length ? [h("h3", {}, "Come sono andate le sestine registrate")] : []),
    ...scored.slice(0, 6).map((e) => h("div", { class: "recent-row" },
      h("span", { class: "when" }, dateIt(e.result.date, { weekday: "short", day: "numeric", month: "short" })),
      ...ORDER.filter((k) => k in e.result.hits).map((k) => h("span", { class: `pill${e.result.hits[k] >= 2 ? " pill--good" : ""}` },
        L.labels[k]?.split(" ·")[0].split(" (")[0] ?? k, " ", h("strong", {}, `${e.result.hits[k]} punti`))))));
}

async function checkFreshness() {
  try {
    const txt = await fetch(REMOTE_DRAWS, { cache: "no-cache" }).then((r) => r.text());
    const lines = txt.trim().split("\n");
    const last = lines[lines.length - 1].split(",");
    const [dd, mm, yyyy] = last[1].split("/");
    const iso = `${yyyy}-${mm}-${dd}`;
    if (iso <= L.data.last) return;
    const nums = last.slice(2, 8).map(Number).sort((a, b) => a - b);
    const fresh = $("#fresh");
    const pick = L.prediction?.target_date <= iso ? L.prediction?.picks.numa : null;
    const hits = pick ? pick.filter((n) => nums.includes(n)).length : null;
    fresh.hidden = false;
    fresh.replaceChildren(h("strong", {}, `Appena uscita, ${dateIt(iso)}: `), nums.join(" "), ` · Jolly ${last[8]}`,
      pick ? h("span", {}, ` · Numa ha fatto ${hits} punt${hits === 1 ? "o" : "i"}.`) : "",
      h("div", { class: "small" }, "Il sito registra il risultato e la prossima sestina al prossimo aggiornamento automatico."));
  } catch (e) { /* offline o fonte non raggiungibile: va bene così */ }
}

// ------------------------------------------------------------ classifica
function renderBoard() {
  const b = L.leaderboard;
  const names = ORDER.filter((k) => b[k]);
  const table = $("#board");
  const head = h("tr", {}, ["Strategia", "Concorsi", "Punti", "Attesi", "Differenza", "p (caso)", "Vincite ipotetiche"]
    .map((t, i) => h("th", { class: i ? "num" : "" }, t)));
  const rows = names.map((k) => {
    const r = b[k];
    const diff = r.hits - r.expected;
    return h("tr", {}, h("td", {}, h("i", { class: "key", style: `background:var(--series-${SERIES[k]})` }), L.labels[k] ?? k),
      h("td", { class: "num" }, r.draws), h("td", { class: "num" }, r.hits), h("td", { class: "num" }, dec(r.expected, 1)),
      h("td", { class: `num${diff > 0 ? " delta-pos" : ""}` }, `${diff >= 0 ? "+" : "−"}${dec(Math.abs(diff), 1)}`),
      h("td", { class: "num" }, r.p_at_least === null ? "–" : dec(r.p_at_least, 2)),
      h("td", { class: "num" }, r.prizes_known ? euro.format(r.prizes_eur) : "–"));
  });
  table.replaceChildren(h("thead", {}, head), h("tbody", {}, rows.length ? rows : h("tr", {}, h("td", { colspan: 7 }, "Ancora nessun concorso verificato."))));
  $("#power-note").textContent = `per distinguere un vantaggio del 10% dal caso servono circa ${num.format(core.drawsForPower(0.1))} concorsi (più di 6 anni). Per un vantaggio del 5%, oltre 25 anni.`;
  const cumulative = names.filter((k) => b[k].draws > 1).map((k) => ({
    label: L.labels[k] ?? k, color: `var(--series-${SERIES[k]})`, values: b[k].cumulative_excess, dates: b[k].dates,
  }));
  mount($("#chart-cumulative"), (c) => lineChart(c, cumulative, { zeroLabel: "come il caso" }));

  const retro = L.retrospective;
  if (retro.backtest_1800) {
    const bt = retro.backtest_1800;
    $("#retro-note").textContent = `Backtest su ${num.format(bt.draws)} concorsi (dicembre 2015 – luglio 2026), ogni previsione fatta solo con i concorsi precedenti. Il caso puro fa in media ${num.format(bt.expected)} punti.`;
    const rows2 = Object.entries(bt.hits).map(([k, v]) => ({ key: k, label: RETRO_NAMES[k] ?? k, value: v,
      tip: `${RETRO_NAMES[k] ?? k}: ${v - bt.expected >= 0 ? "+" : ""}${v - bt.expected} rispetto al caso` }))
      .sort((a, b2) => b2.value - a.value);
    mount($("#chart-retro"), (c) => hbarChart(c, rows2, { ref: bt.expected, refLabel: `caso: ${bt.expected}`, min: 600, highlight: "ensemble temporale calibrato" }));
  }
  if (retro.holdout) {
    const ho = retro.holdout, m = ho.models;
    const ens = m["ensemble temporale calibrato"], gbm = m["gradient boosting"];
    $("#holdout-note").replaceChildren(
      h("p", {}, h("strong", {}, "La prova del nove. "), `Il miglior modello è stato congelato il 30 luglio 2026 e messo alla prova sui ${ho.draws} concorsi successivi (fino al ${dateIt(ho.period[1])}).`),
      h("p", {}, `Ensemble: ${ens.hits} punti, gradient boosting: ${gbm.hits}, contro ${dec(ens.expected, 1)} attesi per puro caso. Nessun vantaggio.`),
      h("p", {}, "I 4 test statistici sullo storico non distinguono le estrazioni da un generatore casuale. Il SuperEnalotto non si prevede: si può solo scegliere come perdere meno."));
  }
}

// ------------------------------------------------------------ popolarità
function heatColor(t) {
  const a = Math.min(1, Math.abs(t));
  const pctA = Math.round(a * 100);
  return t >= 0 ? `color-mix(in oklab, var(--warm-3) ${pctA}%, var(--mid))` : `color-mix(in oklab, var(--cool-3) ${pctA}%, var(--mid))`;
}

function renderPopularity() {
  const theta = model.theta;
  const maxAbs = Math.max(...theta.map(Math.abs));
  $("#heat").replaceChildren(...theta.map((t, i) => {
    const cell = h("div", { class: "cell", style: `background:${heatColor(t / maxAbs)};color:${Math.abs(t / maxAbs) > 0.6 ? "#fff" : "var(--ink)"}`,
      tabindex: 0, "aria-label": `${i + 1}: giocato ${dec(Math.exp(t), 2)} volte la media` }, String(i + 1));
    const show = (e) => showTip(e, h("div", { class: "tv" }, `${dec(Math.exp(t), 2)}×`), h("div", {}, `${the(i + 1)} rispetto a un numero medio`));
    cell.addEventListener("pointermove", show);
    cell.addEventListener("pointerleave", hideTip);
    return cell;
  }));
  const pop = L.retrospective.popularity;
  if (!pop) return;
  const best = Object.entries(pop.strategies).find(([k]) => k.includes("500"))?.[1];
  const r2 = pop.validation["3"];
  $("#pop-tiles").replaceChildren(
    h("div", { class: "tile" }, h("div", { class: "label" }, "Premi incassati in più"), h("div", { class: "value" }, signedPct(best.vs_uniform, 0)),
      h("div", { class: "sub" }, `2, 3 e 4 punti, ${pop.years[0]}–${pop.years[1]}`)),
    h("div", { class: "tile" }, h("div", { class: "label" }, "Probabilità di vincere"), h("div", { class: "value" }, "identica"),
      h("div", { class: "sub" }, `${dec(best.hits_per_ticket, 3)} punti a colonna, come il caso`)),
    h("div", { class: "tile" }, h("div", { class: "label" }, "Il modello spiega"), h("div", { class: "value" }, pct(r2, 0)),
      h("div", { class: "sub" }, "delle differenze tra concorsi, su anni mai visti")));
  const names = { caso: "Sestine a caso", "numa (meno giocata tra 20)": "Numa leggero (1 su 20)",
    "numa (meno giocata tra 500)": "Numa (1 su 500)", "placebo (popolarita' permutata)": "Placebo" };
  const tips = { "placebo (popolarita' permutata)": "Stessa procedura con la popolarità mescolata a caso: nessun effetto" };
  const popRows = Object.entries(pop.strategies).map(([k, v]) => ({ key: k, label: names[k] ?? k, value: v.return,
    tip: tips[k] ?? `${names[k] ?? k}: ${signedPct(v.vs_uniform)} rispetto al caso` }));
  mount($("#chart-pop"), (c) => hbarChart(c, popRows, { format: (v) => euro.format(v), highlight: "numa (meno giocata tra 500)", ref: pop.uniform_return, refLabel: "caso (teorico)" }));
  $("#pop-note").textContent = `Euro incassati in media per ogni euro giocato con premi da 2, 3 e 4 punti, su ${num.format(pop.draws)} concorsi (${pop.years[0]}–${pop.years[1]}), con le quote realmente pagate. Il modello di ogni anno è stimato solo sugli anni precedenti. Si perde comunque, ma meno: il gioco restituisce in premi circa il 60% di quanto incassa.`;
}

// ------------------------------------------------------------ controlla
const picked = new Set();
function setupPicker() {
  const grid = $("#picker");
  for (let n = 1; n <= 90; n++) {
    grid.append(h("button", { type: "button", "aria-pressed": "false", "data-n": n, onclick: () => toggle(n) }, String(n)));
  }
  $("#pick-random").addEventListener("click", () => setPicked(core.randomTicket()));
  $("#pick-numa").addEventListener("click", () => setPicked(core.generate(model, 1, { past, last: lastNums })[0]));
  $("#pick-clear").addEventListener("click", () => setPicked([]));
}
function toggle(n) {
  if (picked.has(n)) picked.delete(n); else if (picked.size < 6) picked.add(n);
  sync();
}
function setPicked(arr) { picked.clear(); arr.forEach((n) => picked.add(n)); sync(); }
function sync() {
  for (const b of $("#picker").children) b.setAttribute("aria-pressed", picked.has(Number(b.dataset.n)) ? "true" : "false");
  const out = $("#check-result");
  if (picked.size < 6) { out.replaceChildren(h("p", { class: "muted" }, `Seleziona ancora ${6 - picked.size} numer${6 - picked.size === 1 ? "o" : "i"}.`)); return; }
  const t = [...picked].sort((a, b) => a - b);
  const rel = core.relativePopularity(t, model);
  const issues = core.patternIssues(t, { past, last: lastNums });
  const hist = core.history(t, D.rows);
  const pos = Math.max(0, Math.min(100, 50 + 50 * Math.log(rel) / Math.log(4)));
  const verdict = rel < 0.7 ? "poco giocata: se vince, dividi con poche persone" : rel > 1.4 ? "molto giocata: se vince, dividi con tante persone" : "giocata più o meno come una sestina qualsiasi";
  const lastHits = t.filter((n) => lastNums.includes(n)).length;
  out.replaceChildren(
    h("h3", {}, t.join(" · ")),
    h("p", {}, `Giocata circa ${dec(rel, 2)}× rispetto a una sestina tipica: ${verdict}.`),
    h("div", { class: "gauge", "aria-hidden": "true" }, h("i", { style: `left:${pos}%` })),
    h("div", { class: "gauge-labels" }, h("span", {}, "rara"), h("span", {}, "tipica"), h("span", {}, "affollata")),
    issues.length ? h("ul", { class: "issues" }, issues.map((x) => h("li", {}, `Attenzione: ${x}.`))) : h("p", { class: "small" }, "Nessuno schema “a disegno”: bene."),
    h("p", { class: "small" }, `All'ultima estrazione (${dateIt(L.last_draw.date)}) avrebbe fatto ${lastHits} punt${lastHits === 1 ? "o" : "i"}.`),
    h("div", { class: "hist" }, [2, 3, 4, 5, 6].map((k) => h("div", {}, h("b", {}, hist.counts[k]), h("span", {}, `${k} punti`)))),
    h("p", {}, `Giocandola a ogni concorso dal 1997 (${num.format(hist.draws)} concorsi) avresti speso ${euro0.format(hist.spent)} e vinto ${euro0.format(hist.won)}`,
      hist.unknown ? `, più ${hist.unknown === 1 ? "un premio" : `${hist.unknown} premi`} di importo non registrato nell'archivio.` : "."),
    hist.notable.length ? h("p", { class: "small" }, "Risultati migliori: ",
      hist.notable.slice(0, 5).map((x) => `${x.hits}${x.jolly ? "+1" : ""} punti il ${dateIt(x.date, { day: "numeric", month: "short", year: "numeric" })}`).join(", "), ".") : null,
    h("p", { class: "small muted" }, "La probabilità che esca è sempre 1 su 622.614.630, qualunque sestina tu scelga."));
}

// ------------------------------------------------------------ statistiche
function renderStats() {
  const st = L.stats;
  const expected = (st.draws * 6) / 90;
  $("#stats-intro").textContent = `${num.format(st.draws)} estrazioni dal ${dateIt(L.data.first)} al ${dateIt(L.data.last)}. Le differenze tra numeri sono quelle che ci si aspetta dal caso: nessun numero è "fortunato".`;
  mount($("#chart-freq"), (c) => columnChart(c, st.frequency, { expected, label: "Uscite di ogni numero da 1 a 90" }));
  const late = st.gap.map((g, i) => [i + 1, g]).sort((a, b) => b[1] - a[1]).slice(0, 10);
  $("#late").replaceChildren(...late.map(([n, g]) => h("li", {}, h("strong", {}, n), ` manca da ${g} concorsi`)));
  mount($("#chart-gap"), (c) => dotChart(c, st.p_exit_given_gap, { ref: 6 / 90 }));
}

function renderMethod() {
  const r = L.retrospective;
  const t = r.tests ?? {};
  const pop = r.popularity;
  const boxes = [
    ["1 · I dati", `${num.format(L.data.draws)} estrazioni dal 1997, verificate su due fonti indipendenti (un errore trovato e corretto, un concorso mancante recuperato). Le quote e i vincitori di ogni categoria vengono da SuperEnalottoOggi.com.`],
    ["2 · Il caso", `Quattro test: frequenze uniformi (p = ${dec(t.uniformity_p ?? 0, 2)}), estrazioni indipendenti (p = ${dec(t.overlap_p ?? 0, 2)}), ritardi senza memoria (p = ${dec(t.gaps_p ?? 0, 2)}), nessuna autocorrelazione (p = ${dec(t.autocorr_p ?? 0, 2)}). Niente da prevedere.`],
    ["3 · I modelli", "Sette modelli, dalla frequenza alla rete neurale LSTM, valutati in walk-forward su 1.800 concorsi: nessuno batte il caso in modo stabile. Un apparente successo (p = 0,008) si è rivelato un falso positivo."],
    ["4 · La popolarità", pop ? `Dai vincitori di 2, 3 e 4 punti si stima quanto è giocato ogni numero (R² ${dec(pop.validation["3"], 2)} su anni mai visti). Giocare numeri poco giocati ha reso il ${signedPct(Object.values(pop.strategies).find((v, i) => Object.keys(pop.strategies)[i].includes("500")).vs_uniform, 0)} sui premi, con un controllo placebo a zero.` : "Modello di popolarità dei numeri dalle quote storiche."],
    ["5 · Il registro", "Ogni previsione è salvata prima della chiusura delle giocate, con l'hash della precedente (catena non modificabile) e il commit automatico su GitHub come seconda prova."],
    ["6 · Il codice", "Python, aperto e riproducibile: ogni numero di questa pagina viene da uno script del progetto. Il sito si aggiorna da solo dopo ogni estrazione."],
  ];
  $("#method").replaceChildren(...boxes.map(([title, text]) => h("div", {}, h("h4", {}, title), text)));
  $("#sources").textContent = `Dati: ${L.data.sources.join(" · ")}. Aggiornato il ${new Date(L.generated_utc).toLocaleString("it-IT", { dateStyle: "long", timeStyle: "short" })}.`;
}

function setupTheme() {
  $(".theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("numa-theme", next); } catch (e) { /* storage non disponibile */ }
  });
}

async function main() {
  setupTheme();
  try {
    await load();
  } catch (e) {
    $("#hero-balls").replaceChildren(h("p", { class: "muted" }, "Non riesco a caricare i dati. Se apri il file direttamente, usa invece: python numa.py"));
    return;
  }
  renderHero();
  setupGenerator();
  setupBubble();
  renderStrategies();
  renderLast();
  renderBoard();
  renderPopularity();
  setupPicker();
  renderStats();
  renderMethod();
  checkFreshness();
}

main();
