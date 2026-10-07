#!/bin/sh
# numa in locale con un solo comando:  ./run_local.sh
# La prima volta prepara l'ambiente Python; poi aggiorna i dati, calcola le
# sestine del prossimo concorso e apre il sito su http://localhost:8765
# Opzioni utili: --no-update (offline), --no-ml (piu' veloce), --port 9000
set -e
cd "$(dirname "$0")"

# ".nosync": iCloud Drive non sincronizza questa cartella e non la sposta nel cloud
# (con la Scrivania su iCloud, un ambiente Python normale diventa lentissimo).
VENV=".venv.nosync"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Prima esecuzione: preparo l'ambiente Python (qualche minuto)..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements-site.txt
fi

EXTRA=""
# Se il progetto e' collegato a GitHub il registro ufficiale lo tiene l'automazione:
# ci si allinea a quello e in locale la sestina si mostra senza registrarla.
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git checkout -- data site/data 2>/dev/null || true   # file generati: si rigenerano
  git pull --ff-only --quiet || echo "(GitHub non raggiungibile: uso i dati locali)"
  EXTRA="--no-register"
fi
exec "$VENV/bin/python" numa.py $EXTRA "$@"
