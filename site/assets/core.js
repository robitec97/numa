// Logica condivisa del sito (anche testata con Node: tests/test_core.mjs).
// Rispecchia src/popularity.py: stesso punteggio, stessi filtri.

export const N_BALLS = 90;
export const N_DRAWN = 6;
export const TOTAL_COMBINATIONS = 622614630;

const PAIRS = [];
for (let i = 0; i < N_DRAWN; i++) for (let j = i + 1; j < N_DRAWN; j++) PAIRS.push([i, j]);

const sorted = (t) => [...t].sort((a, b) => a - b);

// Coppie: consecutive, stessa decina (1-10, 11-20, ...), stessa cifra finale.
export function pairCounts(ticket) {
  const t = sorted(ticket);
  let consec = 0, decade = 0, ending = 0;
  for (const [i, j] of PAIRS) {
    const a = t[i], b = t[j];
    if (b - a === 1) consec++;
    if (Math.floor((a - 1) / 10) === Math.floor((b - 1) / 10)) decade++;
    if (a % 10 === b % 10) ending++;
  }
  return [consec, decade, ending];
}

// log-popolarita' della sestina rispetto a una giocata media.
export function score(ticket, model) {
  const phi = [model.phi.consecutivi, model.phi.stessa_decina, model.phi.stessa_cadenza];
  const pc = pairCounts(ticket);
  let s = 0;
  for (const n of ticket) s += model.theta[n - 1];
  for (let k = 0; k < 3; k++) s += pc[k] * phi[k];
  return s;
}

// Quante volte e' giocata rispetto a una sestina tipica (mediana).
export function relativePopularity(ticket, model) {
  return Math.exp(score(ticket, model) - (model.baseline_median ?? 0));
}

const key = (t) => sorted(t).join("-");

export function pastIndex(rows) {
  const m = new Map();
  for (const r of rows) m.set(key(r[1]), r[0]);
  return m;
}

// Motivi per cui una sestina e' "a disegno" (molto giocata): vuoto se ok.
export function patternIssues(ticket, { past = null, last = null } = {}) {
  const t = sorted(ticket);
  const out = [];
  for (let i = 0; i + 2 < t.length; i++) {
    if (t[i + 1] - t[i] === 1 && t[i + 2] - t[i + 1] === 1) { out.push("tre o più numeri consecutivi"); break; }
  }
  if (hasProgression(t, 4)) out.push("quattro numeri in progressione regolare");
  const decades = new Array(9).fill(0);
  for (const n of t) decades[Math.floor((n - 1) / 10)]++;
  if (Math.max(...decades) >= 4) out.push("quattro o più numeri nella stessa decina");
  if (t.every((n) => n <= 31)) out.push("tutti numeri fino a 31: sembrano date di nascita");
  if (last && t.filter((n) => last.includes(n)).length >= 3) out.push("tre o più numeri dell'ultima estrazione");
  if (past && past.has(key(t))) out.push(`è già uscita come sestina vincente (${past.get(key(t))})`);
  return out;
}

function hasProgression(t, length) {
  const s = new Set(t);
  for (let i = 0; i < t.length; i++) {
    for (let j = i + 1; j < t.length; j++) {
      const step = t[j] - t[i];
      let ok = true;
      for (let k = 2; k < length; k++) if (!s.has(t[i] + k * step)) { ok = false; break; }
      if (ok) return true;
    }
  }
  return false;
}

export function randomTicket(rand = Math.random) {
  const pool = Array.from({ length: N_BALLS }, (_, i) => i + 1);
  for (let i = 0; i < N_DRAWN; i++) {
    const j = i + Math.floor(rand() * (N_BALLS - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return sorted(pool.slice(0, N_DRAWN));
}

// n sestine, ognuna la meno giocata tra `candidates` sestine casuali che
// superano i filtri; sestine diverse condividono al piu' maxOverlap numeri.
export function generate(model, n = 1, { candidates = 500, past = null, last = null, maxOverlap = 2, rand = Math.random } = {}) {
  const chosen = [];
  let guard = 0;
  while (chosen.length < n && guard++ < 50 * n) {
    const pool = Array.from({ length: candidates }, () => randomTicket(rand));
    pool.sort((a, b) => score(a, model) - score(b, model));
    for (const t of pool) {
      if (chosen.some((c) => c.filter((x) => t.includes(x)).length > maxOverlap)) continue;
      if (patternIssues(t, { past, last }).length === 0) { chosen.push(t); break; }
    }
  }
  return chosen;
}

// Storico di una sestina: punti per concorso, premi ipotetici, spesa.
// rows: [data, numeri, jolly, superstar, prezzo, premi[2,3,4,5,5+1,6] | null]
export function history(ticket, rows) {
  const set = new Set(ticket);
  const counts = new Array(7).fill(0);
  let spent = 0, won = 0, best = 0, unknown = 0;
  const notable = [];
  for (const [date, nums, jolly, , price, prizes] of rows) {
    let hits = 0;
    for (const x of nums) if (set.has(x)) hits++;
    counts[hits]++;
    spent += price;
    if (hits >= 2 && prizes) {
      const idx = { 2: 0, 3: 1, 4: 2, 5: set.has(jolly) ? 4 : 3, 6: 5 }[hits];
      const p = prizes[idx];
      if (p === null || p === undefined) unknown++;
      else won += p;
    } else if (hits >= 5) unknown++;
    if (hits > best) best = hits;
    if (hits >= 3) notable.push({ date, hits, jolly: hits === 5 && set.has(jolly) });
  }
  // i risultati migliori prima, a parita' i piu' recenti
  notable.sort((a, b) => b.hits - a.hits || (a.date < b.date ? 1 : -1));
  return { counts, spent, won, best, unknown, notable: notable.slice(0, 8), draws: rows.length };
}

// Concorsi necessari per vedere un vantaggio relativo `edge` sui punti
// (test a una coda, alfa 5%, potenza 80%): varianza ipergeometrica 0,3524.
export function drawsForPower(edge, alphaZ = 1.645, powerZ = 0.8416) {
  const sd2 = (6 * 6 / 90) * (84 / 90) * (84 / 89);
  return Math.ceil(((alphaZ + powerZ) ** 2 * sd2) / (0.4 * edge) ** 2);
}
