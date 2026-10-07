# numa — si può predire il SuperEnalotto con il machine learning?

Esperimento didattico completo: dati reali, modelli veri, valutazione onesta.
**Risposta: no.** Nessun modello batte il caso su dati nuovi. Ma i dati
sulle quote mostrano una cosa che si può fare davvero: **giocare numeri poco
giocati dagli altri** non aumenta la probabilità di vincere, ma aumenta quanto
si incassa quando si vince (**+13% sui premi da 2, 3 e 4 punti**, verificato
walk-forward sulle quote reali 2012–2026, con controllo placebo a zero).

Il progetto ha una mascotte, **Numa**, disegnata a partire da una foto di
Pompilio, a cui è dedicato.

## Avvio rapido

```bash
./run_local.sh          # oppure doppio clic su "Avvia numa.command" nel Finder
```

La prima volta prepara l'ambiente Python in `.venv.nosync` (qualche minuto),
poi aggiorna estrazioni e quote, calcola le sestine del prossimo concorso e
apre il sito su <http://localhost:8765>. Opzioni: `--no-update` (offline),
`--no-ml` (salta i modelli ML), `--port 9000`. Altri comandi:

```bash
.venv.nosync/bin/python numa.py genera 5                  # 5 sestine poco giocate
.venv.nosync/bin/python numa.py controlla 3 7 12 19 25 29 # storico di una sestina
.venv.nosync/bin/python numa.py build                     # solo dati e sito, senza server
```

**Perché `.venv.nosync`:** la Scrivania è sincronizzata con iCloud Drive, che
sposta nel cloud i file poco usati. Il vecchio `.venv` (migliaia di file) era
finito così: ogni `import pandas` riscaricava i file uno per uno e impiegava
minuti. iCloud ignora le cartelle che finiscono in `.nosync`. Il vecchio `.venv`
(con PyTorch, per `run_rnn.py` e `predict_next.py`) è ancora lì.

## Il sito

`site/` è una pagina statica (HTML, CSS, JavaScript senza dipendenze) che legge
`site/data/*.json`, generati da `numa.py build`:

- **i numeri di Numa** per il prossimo concorso, più un generatore di altre
  sestine poco giocate che gira nel browser;
- **la classifica onesta**: sei strategie (Numa, ensemble ML, gradient
  boosting, numeri caldi, ritardatari, caso) registrate *prima* di ogni
  concorso e verificate *dopo*, con p-value esatto e vincite ipotetiche;
- **controlla i tuoi numeri**: quanto è giocata una sestina e cosa avrebbe
  fatto in tutti i concorsi dal 1997;
- ultima estrazione con le quote, statistiche, e i risultati di questa ricerca.

**Aggiornamento automatico:** `.github/workflows/numa.yml` gira dopo ogni
estrazione e ogni mattina su GitHub Actions: aggiorna i dati, verifica le
previsioni, registra la sestina del concorso successivo, fa il commit e
pubblica su GitHub Pages. Se la fonte non ha ancora l'ultima estrazione,
aspetta a registrare (fino a 3 ore prima della chiusura delle giocate). Nota:
GitHub sospende i workflow pianificati dei repository pubblici dopo 60 giorni
senza attività; i commit automatici dovrebbero bastare, altrimenti si
riattiva dalla scheda Actions.

**Il registro** (`site/data/ledger.json`, `src/ledger.py`) è a catena di hash:
ogni voce contiene l'hash della precedente, e il commit automatico su GitHub
ne certifica l'ora. Una voce si aggiunge solo prima delle 19:30 del giorno del
concorso e viene abbinata alla prima estrazione successiva (anche se il
concorso viene spostato). È la verifica prospettica che mancava: qui nessuna
scelta può essere adattata agli esiti. La prima voce è la previsione del
05/09/2026 già presente in `results/predictions/`.

## Dati

4 279 estrazioni dal 3/12/1997 al 6/10/2026 (`data/superenalotto_full.txt`,
da [Lottopyrhon/Estrazioni_Superenalotto](https://github.com/Lottopyrhon/Estrazioni_Superenalotto)),
validate incrociandole con [luigimassa/superenalotto-archivio](https://github.com/luigimassa/superenalotto-archivio):
su 4 183 estrazioni in comune, un solo disaccordo (27/12/2004), risolto con una
terza fonte (superenalotto.net) e corretto in `src/data.py`.

Quote e numero di vincitori di ogni categoria, concorso per concorso dal 1997
(`data/quote_superenalottooggi.csv`): [SuperEnalottoOggi.com](https://superenalottooggi.com/archivio),
licenza CC BY 4.0 (*Dati: SuperEnalottoOggi.com*). Confrontate con
superenalotto.net su 80 concorsi a caso (2009–2026): identiche. I numeri estratti
coincidono con lo storico su tutti i 4 279 concorsi (stesso errore del
27/12/2004, già corretto). Sette errori della fonte, comuni anche a
superenalotto.net, sono corretti a runtime in `src/quote.py` (`FIXES`), che
ricostruisce anche montepremi, colonne giocate e popolarità di ogni concorso
dalle regole di ripartizione (1997, 2008, 2016).

## Pipeline

| Script | Cosa fa |
|---|---|
| `numa.py` | aggiorna dati e quote, registro, sestine del prossimo concorso, sito |
| `run_local.sh` | tutto quanto sopra con un comando, e apre il sito in locale |
| `update_data.py` | scarica lo storico aggiornato e lo valida prima di sovrascrivere |
| `run_stats.py` | 4 test statistici di casualità sullo storico |
| `run_backtest.py` | backtest walk-forward di 6 modelli sugli ultimi 600 concorsi |
| `run_replication.py` | replica su 3 finestre disgiunte + sensibilità al seed |
| `run_rnn.py` | stessa valutazione per la rete ricorrente (LSTM, PyTorch) |
| `run_improvement.py` | ensemble calibrato, tre finestre fisse, inferenza esatta e intervalli appaiati |
| `run_holdout.py` | controlla le estrazioni successive al cutoff con il protocollo congelato |
| `run_popularity.py` | modello di popolarità dalle quote e backtest dei premi reali |
| `make_figures.py` | genera le 5 figure in `figures/` |
| `predict_next.py` | aggiorna i dati e stampa le "previsioni" di GBM e LSTM |

`update_data.update()` scarica il file remoto, ne valida l'integrità (range,
duplicati, date) e verifica che **le estrazioni già note coincidano** con
quelle locali prima di sovrascrivere: se la fonte remota contraddice il
passato, l'aggiornamento viene rifiutato. Stessa regola per le quote
(`src/quote.py`).

```bash
python3 -m venv .venv.nosync && .venv.nosync/bin/pip install -r requirements.txt
.venv.nosync/bin/python run_stats.py && .venv.nosync/bin/python run_backtest.py
.venv.nosync/bin/python run_replication.py && .venv.nosync/bin/python make_figures.py
```

`requirements-site.txt` contiene solo quanto serve al sito (senza PyTorch).

### Modelli confrontati

Il problema è formulato come classificazione binaria per coppia
(concorso *t*, numero *n*): P(*n* esce al concorso *t*) date solo le estrazioni
precedenti a *t* (nessun leakage, verificato con test). Feature: frequenze su
finestre di 10/50/200 concorsi, medie mobili esponenziali, ritardo attuale,
frequenza storica, identità del numero.

1. **caso (uniforme)** — baseline teorico, p = 6/90 per ogni numero
2. **frequenza storica globale** — i numeri usciti più spesso dal 1997
3. **numeri caldi** — i più frequenti nelle ultime 50 estrazioni
4. **ritardatari** — i numeri assenti da più tempo (la strategia da bar)
5. **regressione logistica** — ML lineare sulle feature
6. **gradient boosting** — ML non lineare (HistGradientBoosting)
7. **LSTM** — rete ricorrente (PyTorch) che legge le ultime 50 estrazioni
   come sequenza di vettori binari a 90 dimensioni e produce 90 probabilità
   (BCE multi-label, early stopping su split temporale di validazione)

I modelli ML sono riallenati ogni 30 concorsi su finestra espandente; ogni
concorso viene giocata la sestina dei 6 numeri con punteggio più alto.

## Risultati

### 1. I test eseguiti non rifiutano l'ipotesi di casualità

| Test | p-value | Verdetto |
|---|---|---|
| Uniformità delle frequenze dei 90 numeri (χ² + Monte Carlo) | 0.15 | uniforme |
| Sovrapposizione tra estrazioni consecutive vs ipergeometrica | 0.55 | indipendenti |
| Distribuzione dei ritardi vs geometrica (χ² GOF) | 0.56 | senza memoria |
| Autocorrelazione della somma dei 6 numeri (Ljung-Box, 10 lag) | 0.97 | nessuna |

La figura 2 è il cuore didattico: P(un numero esca | è in ritardo da ≥ g
concorsi) resta inchiodata al 6,67 % teorico per ogni g. **I ritardatari non
"devono" uscire**: è la fallacia del giocatore, misurata su 28 anni di dati.

### 2. Risultati originali: il segnale del gradient boosting

Backtest sugli ultimi 600 concorsi (set 2023 → lug 2026), attesi 240 hit per
puro caso (σ ≈ 14.5):

| Modello | Hit | p(≥ caso) |
|---|---|---|
| caso (uniforme) | 236 | 0.62 |
| freq. storica globale | 250 | 0.26 |
| numeri caldi | 230 | 0.76 |
| ritardatari | 230 | 0.76 |
| regressione logistica | 254 | 0.18 |
| **gradient boosting** | **276** | **0.008** |
| LSTM (rete ricorrente) | 256 | 0.14 |

Il gradient boosting merita un approfondimento, ma il p-value da solo non
stabilisce se il risultato sia un falso positivo. Il ricalcolo per
convoluzione ipergeometrica dà **p = 0,008182**; la correzione Holm sui sei
modelli della tabella originale dà **0,04909**, includendo la LSTM
**0,05727**. Sono famiglie di confronti diverse, e nessuna conteggia tutte
le scelte esplorative fatte nel progetto. Altri due controlli:

- **Replica su finestre disgiunte** — finestra A (2015–2019): 233 hit
  (*sotto* la media!); finestra B (2019–2023): 249 hit (p = 0.28). Il
  "segnale" esiste solo nella finestra C.
- **Sensibilità al seed** — cambiando solo il seed del GBM sulla stessa
  finestra C: 246, 247, 253, 270, 276 hit (p da 0.35 a 0.008). L'"effetto"
  è sensibile alla casualità interna del modello.

Sui 1 800 concorsi il GBM originale totalizza **758 hit contro 720 attesi**,
con p unilaterale esatto **0,06903**. Le repliche non stabiliscono un
vantaggio robusto. Non rifiutare l'ipotesi nulla non equivale a dimostrare
che ogni possibile segnale sia assente.

### 2b. Nemmeno più capacità aiuta: la rete ricorrente

L'LSTM (PyTorch, 50 estrazioni di contesto, early stopping su validazione
temporale) è il test del "serve solo un modello più potente":

- hit nelle 3 finestre: **238, 229, 256** (p = 0.57, 0.78, 0.14) — tutto caso;
- 5 seed sulla finestra C: 240–262 hit (p da 0.51 a 0.07), pura varianza;
- la log-loss di **test** nella finestra C è **0,245104**, contro
  **0,244930** dell'uniforme. Questa specifica rete, con questi dati e
  questo addestramento, non migliora le probabilità della baseline.
  Non è una prova su tutti i possibili modelli o pattern.

### 2c. Nuovo esperimento: ensemble temporale calibrato

Tre GBM con bootstrap per concorso (seed 1, 7, 42), alberi meno profondi e
più regolarizzazione. A ogni riallenamento, gli ultimi 360 concorsi già
osservati sono separati dal training: 240 per l'early stopping temporale e
120 distinti per calibrare il peso dell'ensemble rispetto all'uniforme.
La calibrazione sceglie tra pesi fissi 0/0,1/0,25/0,5/0,75/1 sulla log-loss.
Con peso zero il modello torna all'uniforme; con peso positivo lo shrinkage
riduce la fiducia senza cambiare il ranking dell'ensemble.

| Modello | A: 2015–2019 | B: 2019–2023 | C: 2023–2026 | Totale |
|---|---:|---:|---:|---:|
| Atteso sotto il caso | 240 | 240 | 240 | 720 |
| GBM originale | 233 | 249 | 276 | 758 |
| **Ensemble temporale calibrato** | **256** | **265** | **269** | **790** |

Il nuovo modello ottiene **32 hit in più del GBM (+4,2%)** e 70 in più
dell'attesa casuale (+9,7%). Il p-value nominale del totale è **0,003191**.
I tre periodi erano già stati esplorati: questo resta un risultato di
ricerca retrospettiva, anche se ogni previsione usa soltanto il passato.
Il p-value nominale non corregge le scelte esplorative o i confronti multipli.

La metrica primaria fissata per questa prova è la log-loss rispetto
all'uniforme: **0,24492433 contro 0,24493003**. La differenza media è
−0,00000570, con IC 95% bootstrap a blocchi **[−0,00002672; +0,00001536]**.
Comprende zero: il miglioramento delle probabilità non è conclusivo.
Anche l'IC della differenza di hit rispetto al GBM comprende zero.
Il guadagno in hit non va interpretato come guadagno percentuale nella
probabilità di jackpot o nel rendimento economico.

**Controllo successivo al cutoff (aggiornato al 07/10/2026):** dopo aver
fissato il modello sono state valutate le **39 estrazioni dal 31/07/2026 al
06/10/2026**. L'ensemble ottiene **16 hit**, il GBM 13, contro **15,6 attesi**;
p nominale **0,50** (GBM 0,80). Il peso calibrato è **zero** in entrambi i
riallenamenti: l'ensemble coincide con l'uniforme (anch'essa 16 hit).
Nessun vantaggio sui dati nuovi. (Al 01/09 erano 19 estrazioni e 10 hit.)

Nel frattempo la fonte ha **aggiunto un concorso che mancava**: il
**08/06/2026** (concorso 91/2026, spostato di lunedì), confermato dall'archivio
indipendente delle quote. Lo storico usato a luglio ne era privo: l'effetto sui
risultati è trascurabile (1 concorso su 1 800), ma l'hash del protocollo non
corrisponde più. `run_holdout.py` ora usa lo storico congelato
(`data/superenalotto_cutoff_2026-07-30.txt`, hash verificato) più le sole
estrazioni nuove, e registra l'inserimento in `results/holdout.json`; una
contraddizione dei dati già noti continua a bloccare il controllo. Per
riprodurre esattamente `run_improvement.py` serve quel file congelato.

```bash
.venv.nosync/bin/python run_improvement.py
.venv.nosync/bin/python -m unittest discover -s tests -v
.venv.nosync/bin/python run_holdout.py
```

Richiede scikit-learn >= 1.7 per fornire esplicitamente la validazione
temporale ([API ufficiale](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)).
`results/improvement.json` contiene metriche, audit dei p-value originali,
intervalli e pesi di ogni fit; `.npz` conserva tutte le previsioni.
`.protocol.json` registra configurazione e hash di dati/codice prima del
backtest. La data finale predefinita resta il **30/07/2026** anche dopo un
aggiornamento dei dati. `run_holdout.py` verifica gli hash prima del download
e salva uno storico separato in `data/superenalotto_holdout.txt` e l'esito
in `results/holdout.json`. Ripeterlo sugli stessi dati non aggiunge evidenza
indipendente. `predict_next.py` continua a usare GBM e LSTM originali: il
nuovo modello resta un candidato sperimentale.

I test (ora 26, `python -m unittest discover -s tests`) verificano separazione
temporale, bootstrap per concorso, probabilità valide e inferenza, più
calendario, registro, modello di popolarità e parità tra Python e JavaScript. Corretto anche un caso limite della
normalizzazione che poteva produrre probabilità maggiori di 1.

### 3. Il conto economico originale

Giocando 1 colonna da 1 € a concorso per 600 concorsi (vincite medie storiche
per categoria): tutte le strategie perdono tra il 47 % e il 78 % della spesa.
Il payout del SuperEnalotto restituisce in premi ~60 % del montepremi solo
contando il jackpot, che ha probabilità 1 su 622 614 630 per colonna.

### 4. Sestine impopolari: l'unica ottimizzazione che funziona

P(vincere) non si può alzare, ma i premi di 2, 3, 4 e 5 punti sono un fondo
fisso diviso tra tutti i vincitori. Dal numero di vincitori di ogni concorso,
rispetto a quelli attesi se tutti giocassero a caso (`pop_k`), si legge quanto
erano giocati i numeri estratti. `src/popularity.py` stima un modello
log-lineare: ogni numero ha una popolarità θ, e tre effetti di coppia
(numeri consecutivi, stessa decina, stessa cifra finale). Le costanti che
legano θ a 2, 3 e 4 punti seguono dal modello (k/6 − (6−k)/84), non sono
stimate; un'intercetta per anno e categoria assorbe prezzi e regole.

- **Fuori campione** (stima 2002–2019, verifica 2020–2026, dentro l'anno):
  R² **0,80** su log pop₃, **0,80** su pop₂, **0,69** su pop₄.
- **I più giocati:** 9, 11, 8, 10, 7, 12, 19, 6, 3, 5, 4, 90, 17, 1, 2:
  i mesi, le date e il 90. **I meno giocati:** 61, 62, 60, 76, 34, 84, 46,
  78, 43, 32, 64, 59, 79, 41, 35. I numeri consecutivi sono *evitati*
  (φ = −0,13), stessa decina e stessa cadenza sono cercate (+0,08, +0,07).
- Il generatore prende la sestina meno giocata tra 500 sestine casuali, dopo
  aver scartato gli schemi che il modello non può misurare (tre consecutivi,
  progressioni, quattro nella stessa decina, tutte ≤ 31, sestine già uscite):
  in mediana è giocata **0,41 volte** una sestina tipica.

**Backtest dei premi** (`run_popularity.py`): per ogni anno dal 2012 il
modello è stimato solo sugli anni precedenti; 20 000 colonne per strategia e
anno, pagate con le quote reali di ciascuno dei 2 462 concorsi.

| Strategia | € per € giocato (2, 3, 4 punti) | vs caso | IC 95% (€/€) |
|---|---:|---:|---|
| Caso (teorico esatto: 0,3383) | 0,3373 | −0,3% | [−0,002; +0,001] |
| Numa, meno giocata tra 20 | 0,3719 | **+9,9%** | [+0,025; +0,043] |
| Numa, meno giocata tra 500 | 0,3824 | **+13,0%** | [+0,027; +0,068] |
| Placebo (θ permutato) | 0,3396 | +0,4% | [−0,017; +0,016] |

Punti per colonna identici (0,400–0,402): cambia solo la quota incassata.
Per categoria: 2 punti +5%, 3 punti +21%, 4 punti +40% (quota media
quando si vince: 29,3 € contro 24,8 € per il 3; 465 € contro 354 € per il 4).
L'effetto cresce con la selettività (20 → 500) e sparisce col placebo.
5 e 5+1 sono troppo rari per una stima diretta (e il jackpot è escluso): il
modello prevede un effetto più forte, ma questi dati non lo misurano.
**Si perde comunque**: anche con Numa la resa resta lontana da 1 € per €.
Importi lordi; le quote di 3 e 4 punti includono i contributi aggiuntivi.

### 5. Il registro prospettico

Dal 07/10/2026 ogni concorso ha sei sestine registrate in anticipo (vedi
*Il sito*). Per distinguere dal caso un vantaggio del 10% sui punti servono
circa 1 360 concorsi (6–7 anni); per il 5%, oltre 25 anni. Il registro serve
soprattutto a mostrare, concorso dopo concorso, che nessuno lo batte.

## Cosa possiamo concludere

Sotto l'ipotesi di estrazioni uniformi e indipendenti, ogni sestina scelta
dal passato ha la stessa distribuzione degli hit. I test sullo storico
non hanno rifiutato questa ipotesi, ma non dimostrano l'assenza di qualsiasi
distorsione o dipendenza: servirebbe anche quantificare la potenza dei test.
L'ensemble calibrato, migliore nel backtest retrospettivo, sui 39 concorsi
successivi al congelamento coincide con il caso (peso calibrato zero).

Quello che si può migliorare non è *se* si vince ma *quanto*: la popolarità
dei numeri è stabile e misurabile, e scegliere quelli poco giocati ha reso il
13% in più sui premi minori, con un controllo placebo pulito. Un p-value non
misura la probabilità che il caso sia la spiegazione vera
([dichiarazione ASA](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf)).

## Cosa si impara (la parte che vale)

- formulazione di un problema di previsione come classificazione per-evento
- feature engineering temporale senza leakage
- backtest walk-forward con retraining periodico
- baseline obbligatori e test d'ipotesi contro il caso (Monte Carlo ed esatti)
- il trittico overfitting / seed sensitivity / confronti multipli
- replica out-of-sample come unico vero arbitro, e la preregistrazione
- cambiare domanda: dai numeri che escono ai numeri che giocano gli altri

## Come riprendere la ricerca

Stato al 07/10/2026: dati, quote, registro e sito aggiornati; 26 test verdi.

1. `./run_local.sh` (o `numa.py build`) aggiorna tutto e verifica che giri.
   Punti fragili: gli URL delle due fonti (`update_data.py`, `src/quote.py`).
2. I numeri citati qui sono riproducibili con la pipeline in tabella; i
   risultati grezzi sono in `results/*.json`, le figure in `figures/`.
3. Per l'ensemble: `results/improvement.json` e `results/holdout.json`.
   Riprovare modelli sulle stesse finestre è esplorazione, non conferma: il
   registro prospettico è il posto giusto per i nuovi candidati.

### Direzioni aperte

- **Jackpot e 5 punti** — il modello prevede un vantaggio maggiore dove i dati
  sono pochi; si potrebbe stimare con una verosimiglianza di Poisson su
  vincitori di 5 e 5+1 (decine per concorso).
- **Schedina** — gli schemi geometrici sul modulo di gioco (righe, colonne,
  diagonali) sono noti in letteratura ma non ancora modellati.
- **SuperStar** — mai analizzato: stessa analisi di casualità e popolarità.
- **Test di casualità più severi** — batterie NIST/Dieharder adattate a
  sequenze ipergeometriche, e potenza dei test.
