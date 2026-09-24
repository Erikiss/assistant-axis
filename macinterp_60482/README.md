# State 60482 — Hypothesen-Triage

**Welches publizierte Paper erklärt den Effekt rund um `state_60482` — und erklärt ihn
überhaupt eines?**

Eine Pile-Passage kam bei Optimizer-Schritt **131278** ins Training von Pythia-1.4B. Über die
Checkpoints, die diesen Schritt einrahmen, stieg die Wahrscheinlichkeit eines zurückgehaltenen
Zieltokens — `" per"` (ID 591), in „wind speeds could reach approximately 74 km **per** hour" —
um relativ +14 %. Das wurde zunächst als **Memorierung** gelesen.

Dieses Projekt prüft diese Lesart. Jede Kandidatenerklärung wird zu einer Probe mit einer
Entscheidungsregel, die vor den Daten feststeht, gegen 40 Kontrollpassagen und zwei
Placebo-Grenzen. Läuft auf einer Colab-A100 in etwa 35–50 Minuten.

> Prosa auf Deutsch, Code und Docstrings auf Englisch — wie in den bisherigen Läufen dieser
> Reihe.

---

## Das Messobjekt

| | |
|---|---|
| Modell | `EleutherAI/pythia-1.4b` (**nicht** `-deduped`, verifiziert), 24 Schichten, 16 Köpfe |
| Checkpoints | step130000, step131000, step132000, step133000 |
| Expositionsgrenze | 131000 → 132000 (Anker trainiert bei 131278) |
| Placebo-Grenzen | 130000 → 131000 und 132000 → 133000 |
| Anker | `state_60482`, 207 Kontexttokens, Ziel `" per"` (591) |
| Name | „McQuarrie" = IDs 3044/3864/274/6595 auf Position 124–127 |
| Kontrollen | 40 Pile-Passagen, Seed 42 |
| Varianten | 10 nie trainierte Schreibvarianten des Namens |

### Was bereits gemessen wurde

Diese Zahlen sind in `m60482/config.py` eingefroren; ein neuer Lauf wird gegen sie geprüft.

| Checkpoint | p(` per`) | Rang | Abstand zum Nachbarn | Top-1 |
|---|---|---|---|---|
| 130000 | 0.09402 | 3 | 0.0528 (56 %) | `ph` 0.726 |
| 131000 | 0.09087 | **2** | **0.00122 (1.3 %)** | `ph` 0.798 |
| 132000 | 0.10461 | 3 | 0.0120 (11 %) | `ph` 0.744 |
| 133000 | 0.10833 | 3 | 0.0961 (89 %) | `ph` 0.638 |

Der einzige Checkpoint, an dem das Ziel Rang 2 hielt, ist zugleich der einzige mit einem
hauchdünnen Abstand — 1.3 % seiner eigenen Wahrscheinlichkeit. Ein Rang ohne seinen Abstand ist
keine Messung (arXiv:2411.00640); `aux.measure_jitter` misst die Lauf-zu-Lauf-Streuung, und
`rank_attribution` erklärt einen Rang für **unaufgelöst**, wenn der Abstand darunter liegt.

Drei Dinge fallen daran auf, und sie tragen das ganze Projekt:

1. **Das Ziel gewinnt nie.** Top-1 ist durchgehend `" ph"` — das Modell schreibt „74 kmph",
   nicht „74 km per hour". Jede publizierte Definition wortgetreuer Memorierung ist über die
   Greedy-Fortsetzung formuliert. Unter keiner davon ist diese Passage memoriert.
2. **Der Rangwechsel 2 → 3 geht in die falsche Richtung.** Die Wahrscheinlichkeit des Ziels
   *stieg* an der Expositionsgrenze (0.09087 → 0.10461). Den Rang verlor es an den Konkurrenten
   `"/"`, der 0.1468 → 0.0897 → 0.1166 → 0.2119 schwankt — an den Placebo-Grenzen stärker als
   an der Expositionsgrenze. Die Schlagzeilen-Statistik misst den Konkurrenten.
3. **Die nie trainierten Namensvarianten sagen dasselbe voraus.** Alle zehn liefern dieselben
   Top-3 (`ph` ≈ 0.744–0.751 / `/` / ` per`) und denselben Zielrang. Den Namen zu zerstören
   kostet nichts — es wird nichts passagenspezifisches adressiert.

Dazu die Attention: ab Schicht 3 liegen 0.17–0.69 der Masse der letzten Position auf
**Position 46, einem Newline** 160 Tokens vor dem Ende; Position 0 bekommt weitere 0.19–0.31.
Die Namensposition bekommt praktisch nichts. Und die Verschiebung dieser Attention an der
Expositionsgrenze ist *kleiner* als bei allen 40 Kontrollen (TV 0.01723 gegen Median 0.03282,
p = 1.000).

---

## Die sechs Paper der Liste

Alle sechs existieren; die Zitate wurden einzeln verifiziert. Das ist nicht selbstverständlich:
Die Liste kam von einem Assistenten, der die Quell-PDF nicht lesen konnte und aus einer
mündlichen Beschreibung rekonstruiert hat.

| Paper | Zitat | Warum es aufkam | Ergebnis |
|---|---|---|---|
| Cliff Tokens | arXiv:2606.25524 | ein einzelnes Token trägt einen großen Effekt | **Gegenstand fehlt** — braucht eine generierte Kette mit prüfbarer Antwort; hier wird *ein* Token teacher-forced bewertet |
| SOPHIA / Self-Loops | arXiv:2607.18100 | Homonym | **Gegenstand fehlt** — SOPHIAs „state" ist ein k-Means-Cluster über Reasoning-Schritten, `state_60482` ein Passagenname aus dem K8-Schema |
| Repeat Curse | arXiv:2504.14218 | Wiederholung war eine Hypothese | **Gegenstand messen** — die Probe misst, ob der Kontext überhaupt wiederholt |
| Verbatim Memorization Circuits | arXiv:2506.21588 | die Gegenprüfung zur Memorierungs-Lesart | **Eintrittskriterium nicht erfüllt** — verlangt Memorierungsscore 1.0, also argmax = Ziel |
| Focus Directions | arXiv:2503.23306 | contextual heads statt bloßer Verhaltensbeobachtung | **prüfbar** — liest irgendein Kopf die Namensstelle? |
| Lost in the Middle at Birth | 2026 (Zitat vor Gebrauch prüfen) | leitet Positionsbias aus der Architektur ab, zeigt ihn an *untrainierten* Netzen | **prüfbar, und das einzige mit erfüllten Voraussetzungen** |

### Die Kandidaten, die nicht auf der Liste stehen

**Attention Sinks** (arXiv:2309.17453 StreamingLLM, arXiv:2402.17762 Massive Activations).
Die größte einzelne Zahl im ganzen Datensatz ist 0.44 Attention-Masse auf einem Newline, und
die Liste sagt dazu nichts. Diese Arbeiten sagen den gemessenen Schichtsplit direkt voraus:
lokale Attention in den untersten Schichten, Sink darüber — gemessen sind Schichten 0–2 lokal
auf `" km"`/`" 74"`, Schichten 3–19 auf nicht-inhaltlichen Positionen. Pythia packt Pile-
Dokumente ohne BOS je Sequenz, weshalb Position 0 hier das Wortfragment `" st"` ist
(arXiv:2504.02732).

**Der eingefrorene Sink** (arXiv:2410.10781). Das einzige Paper, das das *Vorzeichen* des
seltsamsten Befunds vorhersagt: Der Anker verschiebt sich *weniger* als alle 40 Kontrollen.
Wenn der Sink in den ersten paar tausend Schritten entsteht und danach festliegt, ist das
Profil bei Schritt 130000 von 143000 seit ~128000 Schritten gesättigt — zwei Checkpoints im
Abstand von 1000 Schritten können dann keine einzelne Exposition kodieren. Der Attention-Null
folgt dann aus der Checkpoint-Wahl, nicht aus der Abwesenheit eines Effekts.

**Attention ist keine Attribution** (arXiv:2004.10102). Der Grund, warum der Attention-Null
bisher *gar nicht lesbar* ist. In den Residualstrom fließt `α·v`, und Sink-Positionen sind
genau die mit ausgetrockneten Value-Vektoren — so funktioniert ein Sink als No-Op. Eine
Position kann 0.69 der Masse halten und fast nichts beitragen. Ein Null in einer Größe, die
keinen Beitrag misst, ist in beide Richtungen uninformativ. `norm_attribution` rechnet das
Profil als `|α_j|·‖v_j‖` neu.

**Die Metrik selbst** (arXiv:2304.15004). Beide Schlagzeilen-Statistiken sind
unstetig: ein Rang, und eine Wahrscheinlichkeit, gemessen während der dominante Token von
0.798 auf 0.744 fällt. Ein Softmax koppelt alle Tokens; fällt der größte, gewinnen alle
anderen, ohne dass ihnen etwas passiert wäre. Der saubere Test ist ein paarweiser
Logit-Kontrast, denn `log p_a − log p_b = logit_a − logit_b` exakt — der Normierer kürzt sich.

Aus den bereits gemessenen Zahlen gerechnet:

| Grenze | | Δ log p(` per`) | Δ [logit(` per`) − logit(`ph`)] |
|---|---|---|---|
| 130000 → 131000 | Placebo | −0.034 | −0.129 |
| 131000 → 132000 | **Exposition** | **+0.141** | **+0.211** |
| 132000 → 133000 | Placebo | +0.035 | +0.189 |

Im Wahrscheinlichkeitsraum sieht die Expositionsgrenze **4.0-fach** so groß aus wie die größte
Placebo-Grenze. Im Logit-Kontrast nur noch **1.1-fach**. Der Großteil des scheinbaren Effekts
ist Umnormierung.

**Rekonstruktion statt Rekollektion** (arXiv:2406.17746). Die Arbeit trennt Memorierung in
*recitation* (duplizierter Text), *reconstruction* (vorhersagbare Vorlagen) und *recollection*
(seltener Einzelkontakt). „74 km per hour" ist eine Einheiten-Vorlage, die der Korpus überall
liefert. Es ist die einzige Erklärung, die das Varianten-Ergebnis **vorhergesagt** statt nur
geduldet hat: Ist die Fortsetzung rekonstruktiv, darf das Zerstören des Namens nichts kosten.

**Optimierer-Rauschen** (arXiv:2207.00099). Die Drift eines einzelnen Beispiels zwischen zwei
Checkpoints wird von den übrigen ~1.02 Millionen Sequenzen im Fenster getrieben, nicht von der
einen interessanten Passage. Das sagt eine diffuse, nicht lokalisierte Störung voraus — genau
was die Tokenebene-Auswertung gefunden hat.

**Der lokale Fortsetzungsprior.** Die drei wahrscheinlichsten Fortsetzungen sind genau die
drei Schreibweisen der Einheit: `ph`, `/`, ` per`. Das sieht nach einem Häufigkeitswettbewerb
zwischen Einheitenkonventionen aus, und es könnte das ganze Phänomen sein. Die
Trunkierungsleiter entscheidet: Wenn p(` per`) aus den letzten zwei Tokens schon auf dem
Vollkontext-Wert liegt, erklärt jede passagenbezogene Hypothese etwas, das es nicht gibt.

**Nullkalibrierung** — was zwischen zwei Checkpoints passiert, wenn nichts passiert ist. Keine
Erklärung, sondern die Vorbedingung dafür, dass es etwas zu erklären gibt; im Bericht getrennt
ausgewiesen.

---

## Benutzung

### Colab A100

`notebooks/colab_a100_state60482.ipynb` öffnen, Runtime auf A100 stellen, durchlaufen lassen.

### Kommandozeile

```bash
# voller Lauf: Hauptmessung + Trunkierungsleiter + untrainierte Basislinie
#              + norm-gewichtete Attention
python -m m60482.run --passages /pfad/zu/passagen.json --output runs --check-variant

# auf einer kleinen Platte: jeden Checkpoint nach Gebrauch loeschen
python -m m60482.run --passages ... --free-disk

# ohne GPU und ohne Modell: Proben gegen ein Bundle mit den echten Ankerzahlen
python -m m60482.run --smoke

# nur bestimmte Proben, auf einem bereits gemessenen Bundle
python -m m60482.run --reuse-bundle runs/<id>/bundle.npz --probes noise_floor context_dependence
```

Die Zusatzmessungen sind standardmäßig **an**. Ohne sie melden vier Proben `NOT_RUN` —
darunter `context_dependence`, die die Frage am ehesten entscheidet. Jede Stufe wird ins
gecachte Bundle geschrieben, sobald sie fertig ist, also überlebt ein abgestürzter Lauf das,
was er schon gemessen hat.

### Tests

```bash
pytest tests/ -q                          # 60 Tests, keine GPU, kein Download, kein Netz
M60482_NETWORK_TESTS=1 pytest tests/ -q   # + 3 Tests gegen huggingface.co (~12 KB)
```

---

## Aufbau

```
m60482/
  config.py            eingefrorene Fakten + Referenzwerte der früheren Läufe
  passages.py          laden, verifizieren, vier Herkunftsstufen
  pile.py              den Anker exakt aus dem Pile holen (4 KB Range-Request)
  model.py             Checkpoint-Laden, fp32, eager attention, Determinismus
  measure.py           Bundle: alles, was ein Forward-Pass je Passage liefert
  aux.py               Trunkierungsleiter, Greedy-Fortsetzung, Namensersetzung, untrainiert
  stats.py             kontrollbezogene p-Werte und die Grenzen, die darin stehen
  registry.py          der Proben-Vertrag
  papers.py            die Kandidaten, woher jeder kam, und die Polarität je Probe
  report.py            Urteil je Paper, nicht je Probe
  run.py               Treiber
  probes/              eine Datei je Probe
```

### Der Proben-Vertrag

Eine Probe ist eine reine Funktion eines `Bundle`. Sie lädt kein Modell, macht keinen
Forward-Pass, geht nicht ins Netz. Sie beantwortet eine Frage, gegen eine Regel, die vor den
Daten feststand, und sie sagt, was sie **nicht** entscheiden kann — das Feld `cannot_conclude`
ist Pflicht und das Anlegen eines Ergebnisses ohne es schlägt fehl.

Urteile: `SUPPORTED`, `REFUTED`, `INCONCLUSIVE`, `SCOPE_FAILED`, `NOT_RUN`, `ERROR`.
`SCOPE_FAILED` heißt nicht, dass ein Paper falsch ist — es heißt, dass sein Gegenstand hier
nicht vorkommt.

**Polarität.** Eine Probe ist eine *Messung*, und Paper sind sich uneinig, wie sie ausgehen
sollte. Die Varianten-Messung ist der klare Fall: Die Memorierungs-Schaltkreis-Arbeit sagt
voraus, dass die Varianten sich unterscheiden; die Rekonstruktions-Arbeit sagt voraus, dass sie
es nicht tun. Beide Vorhersagen betreffen dieselben Zahlen. Deshalb gehört die Polarität zur
*Paarung*, nicht zu einer Seite: `papers.py` bildet Paper auf `{probe: ±1}` ab, und der Bericht
weist aus, welche Proben für welches Paper invertiert gelesen werden. `SCOPE_FAILED` und
`NOT_RUN` kippen nie — „der Gegenstand fehlt" und „nicht gemessen" sind für niemanden ein Beleg.

### Reihenfolge

Die billigen Proben laufen zuerst, weil sie die Prämisse kippen können. Ist das Zieltoken nie
die Greedy-Fortsetzung, beantworten die teuren mechanistischen Proben eine Frage, die niemand
stellen sollte.

| # | Probe | Frage |
|---|---|---|
| 10 | `memorization_entry` | Ist das Ziel je die Greedy-Fortsetzung? |
| 12 | `multiplicity_ledger` | Wie viele der 19 States zeigen das Muster, und wie viele sollten es zufällig? |
| 15 | `noise_floor` | Überschreitet die Änderung das Checkpoint-Rauschen? |
| 18 | `softmax_renormalization` | Überlebt der Effekt die Messung als Logit-Kontrast? |
| 20 | `rank_attribution` | Bewegte sich das Ziel oder sein Konkurrent? |
| 25 | `name_variant_equivalence` | Sagen nie trainierte Schreibvarianten dasselbe voraus? |
| 30 | `context_dependence` | Wie kurz darf der Kontext werden? |
| 35 | `ctx_nll_structure` | Hat der NLL-Rückgang irgendeine Positionsstruktur? |
| 40 | `attention_sink` | Ist das Profil ein Sink-Profil, und unterscheidet es sich von Kontrollen? |
| 42 | `sink_stability` | Bewegt sich das Profil zwischen benachbarten Checkpoints überhaupt? |
| 45 | `norm_attribution` | Überlebt der Sink die Gewichtung mit dem, was er trägt? |
| 50 | `focus_directions` | Liest irgendein Kopf die Namensstelle? |
| 60 | `position_bias` | Teilen alle 69 Passagen ein Profil? Ist es schon untrainiert da? |
| 70 | `cliff_token_scope`, `self_loop_scope`, `repeat_curse_scope` | Ist der Gegenstand überhaupt vorhanden? |

---

## Zwei technische Festlegungen, die keine Stilfragen sind

**fp32 mit TF32 aus.** Bei step131000 liegen Ziel (0.09087) und Konkurrent `"/"` (0.08965)
0.00122 auseinander — ein Logit-Abstand von `ln(0.09087/0.08965) = 0.0135` nats. Bei realistischen
Logit-Beträgen (|logit| ≈ 10–20) ist das **weniger als ein ulp in fp16 und weniger als ein ulp in
bf16**, und etwa ein ulp in TF32. In fp32 mit TF32 aus sind es ~7000 ulp. Die
Schlagzeilen-Statistik ist ein *Rang*; in den kleineren Formaten ist diese Ordnung nicht
reproduzierbar. fp32-Gewichte sind ~5.7 GB; eine A100-40GB hat reichlich Platz.

Batching verändert Logits übrigens auch — andere Batch-Formen wählen andere cuBLAS-Kernel und
Reduktionsreihenfolgen —, aber in fp32 um ~1e-5, also rund 1000-fach unter dem Abstand. Der Rang
kippt davon nicht. Die Messung läuft trotzdem ungebatcht: 69 × 207 Tokens sind billig genug,
dass Batching nichts kauft, was das Risiko wert wäre.

**`attn_implementation="eager"`.** Unter SDPA und FlashAttention gibt `output_attentions=True`
stillschweigend `None` zurück. Dann fehlen die Attention-Zahlen, statt falsch zu sein — was
schlimmer ist, weil es keinen Fehler auslöst. `model.forward_pass` prüft das und wirft.

---

## Der Anker lässt sich exakt rekonstruieren

`global_sample_index` ist suite-spezifisch, und die Architektur entscheidet nichts: Die Configs
von `pythia-1.4b` und `pythia-1.4b-deduped` sind byteidentisch, und `tokenizer.json` hat auf
beiden Repos und in jeder Revision denselben SHA-256. Nur die Datenreihenfolge unterscheidet sie.

Die ist abfragbar. Der preshuffled-Korpus ist ein flaches `uint16`-Array aus
143000 × 1024 Zeilen zu je 2049 Tokens, ohne Header und ohne Padding — die Arithmetik geht exakt
auf:

```
143000 · 1024 · 2049 · 2 = 600 078 336 000 = Summe der 21 .bin-Shards
600 078 336 000 / 4098   = 146 432 000     = 143000 · 1024      (ohne Rest)
```

Zeile *i* liegt also bei Byte *i* · 4098, und ein einzelner HTTP-Range-Request von 4098 Bytes
holt sie. Kein 602-GB-Download, kein `.idx`.

```python
from m60482 import pile
pile.identify_variant()   # -> 'pythia-1.4b'   (~8 KB)
ctx, target = pile.rebuild_anchor()   # 207 Kontexttokens + 591
```

Ergebnis, nachgeprüft: In `pile-standard-pythia-preshuffled` trägt Zeile 134428942 die Namens-IDs
auf 124–127, `" per"` auf 207, `"\n"` auf 46 und `" st"` auf 0 — der Text endet auf
*„…wind speeds could reach approximately 74 km per hour."*. Im deduplizierten Korpus steht an
derselben Stelle ein völlig anderes Dokument. Das ist die Variante, aus den Artefakten
entschieden.

Praktische Folge: **Der Anker ist `row[:207]`**, hängt also an keinem Drive-Ordner. Nur die
40 Kontrollen brauchen noch die Seed-42-Ziehung aus `pile-uncopyrighted`.
`passages.verify_anchor_against_pile()` prüft einen geladenen Satz dagegen.

## Colab: der Platz, nicht die GPU

Vier fp32-Checkpoints sind **22.6 GB** Safetensors im HF-Cache, und HuggingFace teilt
Blobs zwischen Revisionen desselben Repos **nicht** — jedes `stepNNNNNN` ist eine
vollständige Kopie. Auf Colab füllt sich die Platte vor der GPU, meist beim vierten
Checkpoint, nachdem der Lauf schon vierzig Minuten gekostet hat.

- Erste Zelle: `!df -h /`, und mindestens 30 GB frei verlangen.
- `model.checkpoint(step, free_disk=True)` löscht die Revision nach Gebrauch.
- `HF_HOME=/content/hf_home` setzen, damit der Cache dort liegt, wo man ihn sieht.
- **Den HF-Cache niemals auf Drive zeigen lassen.** Freies Drive hat 15 GB und stirbt beim
  dritten Checkpoint; es ist außerdem langsam für große sequentielle Schreibvorgänge. Drive
  nur für die kleinen Ausgaben mounten: das Bundle (~90 MB für alle vier Checkpoints), den
  Bericht, das Manifest.

VRAM ist unkritisch: Gewichte 5.7 GB, Spitze bei Batchgröße 1 mit sofort reduzierter
Attention etwa 7 GB. Der volle `[layer, head, T, T]`-Tensor ist 66.5 MB je Passage und wird
innerhalb der Schleife auf die letzte Query-Zeile reduziert (312 KB je Passage); alle 69
gleichzeitig zu behalten wären 4.6 GB und wäre sinnlos.

## Wenn die kanonischen Passagen fehlen

Die ursprüngliche `passagen.json` hängt an Exp-2-Tabellen im Drive. `passages.resolve()` sucht
sie, prüft den kanonischen SHA-256 und stuft das Ergebnis ein:

| Stufe | Bedeutung | Ergebnisse vergleichbar? |
|---|---|---|
| `canonical` | Hash stimmt mit dem K8-Snapshot | ja |
| `regenerated` | aus den Exp-2-Tabellen erzeugt, Hash stimmt | ja |
| `reconstructed` | Struktur stimmt, Hash nicht | **nein** — Kontrollen neu gezogen |
| `synthetic` | bedeutungslos | **nein** — nur Pipeline-Test |

Alles unterhalb von `regenerated` markiert jedes nachgelagerte Ergebnis dauerhaft, im Bundle
und im Bericht. `--smoke` benutzt ein Bundle, in dem die *echten* Ankerzahlen stecken und alles
andere gewürfelt ist: gut genug, um die Proben zu entwickeln und vorzuführen, nie für eine
Aussage.

---

## Grenzen, unabhängig vom Ergebnis

- Mit 40 Kontrollen ist der kleinste erreichbare p-Wert **1/41 = 0.0244**. 19 States × 3 Grenzen
  sind 57 Vergleiche; Bonferroni verlangt dafür α = 0.00088. **Der Boden liegt 28-fach darüber.**
  In diesem Design kann kein Ergebnis eine Korrektur überleben — das ist eine Eigenschaft des
  Designs, nicht der Daten, und keine Sorgfalt in der Auswertung repariert es. Nur mehr
  Kontrollen oder ein einziger vorregistrierter Vergleich würden es.
- Unter der Nullhypothese landet die größte der drei Grenzänderungen eines States mit
  Wahrscheinlichkeit 1/3 auf seiner Expositionsgrenze. Bei 19 States sind das ~6.3 erwartete
  Treffer und P(mindestens einer) ≈ 1.0. Ein State, der **ausgewählt wurde, weil** er dieses
  Muster zeigt, belegt damit zunächst nichts.
- Drei Grenzen, eine davon die Expositionsgrenze: Die beiden Placebo-Grenzen sind die gesamte
  Nullverteilung. „Außerhalb des Placebo-Bereichs" heißt „außerhalb eines aus zwei Zahlen
  geschätzten Bereichs".
- Der Anker wurde ausgewählt, **weil** er auffällig war. `stats.selection_note()` zwingt jede
  Probe zu sagen, ob ihre Statistik eine Funktion dieses Auswahlkriteriums ist.
- Attention zeigt, welche Schicht welche Stelle liest, nicht in welcher Reihenfolge. Ein
  Transformer hat Tiefe, keine Zeit.
- Aufmerksamkeitsmasse ist keine Attribution. `norm_attribution` gewichtet mit `‖v‖` und ist
  ein besserer Näherungswert, aber keine Messung des Beitrags: Es ignoriert Auslöschung
  zwischen Positionen. Das direkt zu klären bräuchte Ablation oder Patching; keine dieser
  Proben tut das.
- Die Lesart „das Profil ist eingefroren" ist ein Schluss aus den gemessenen Checkpoints, kein
  Test der Entstehungsaussage. Sie zu bestätigen hieße, frühe Checkpoints zu messen
  (step1000–step10000), wo der Sink entstehen soll. Diese Suite misst vier Checkpoints um
  Schritt 130000.
- Für Pythia-**1.4B** existiert öffentlich **kein** brauchbares SAE. EleutherAIs eigene
  `sparsify`-SAEs decken 70m, 70m-deduped, 160m, 160m-deduped und 410m ab — nichts bei 1B oder
  1.4B. Zwei HuggingFace-Repos nennen `pythia-1.4b` im Namen und sind leere Hüllen (nur
  `.gitattributes`, keine Gewichte). Die vorhandenen sind zudem auf dem finalen Checkpoint
  trainiert. Die Methode des Repeat-Curse-Papers
  ist hier also nicht ausführbar; ein Final-Checkpoint-SAE auf ein step131000-Modell anzuwenden
  würde ein anderes Objekt messen.
