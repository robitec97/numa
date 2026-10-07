// Confronta site/assets/core.js con i valori calcolati in Python (test_site.py).
import { readFileSync } from "node:fs";
import * as core from "../site/assets/core.js";

const fx = JSON.parse(readFileSync(process.argv[2], "utf8"));
const model = { ...fx.model, phi: fx.model.phi };
const mismatches = { score: 0, ok: 0, ok_last: 0, pairs: 0 };
fx.tickets.forEach((t, i) => {
  if (Math.abs(core.score(t, model) - fx.score[i]) > 1e-4) mismatches.score++;
  if ((core.patternIssues(t).length === 0) !== fx.ok[i]) mismatches.ok++;
  if ((core.patternIssues(t, { last: fx.last }).length === 0) !== fx.ok_last[i]) mismatches.ok_last++;
  if (core.pairCounts(t).join() !== fx.pairs[i].join()) mismatches.pairs++;
});
console.log(JSON.stringify(mismatches));
