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
| Modell | `EleutherAI/pythia-1.4b`, 24 Schichten, 16 Köpfe |
| Checkpoints | step130000, step131000, step132000, step133000 |
| Expositionsgrenze | 131000 → 132000 (Anker trainiert bei 131278) |
| Placebo-Grenzen | 130000 → 131000 und 132000 → 133000 |
| Anker | `state_60482`, 207 Kontexttokens, Ziel `" per"` (591) |
| Name | „McQuarrie" = IDs 3044/3864/274/6595 auf Position 124–127 |
| Kontrollen | 40 Pile-Passagen, Seed 42 |
| Varianten | 10 nie trainierte Schreibvarianten des Namens |

### Was bereits gemessen wurde

Diese Zahlen sind in `m60482/config.py` eingefroren; ein neuer Lauf wird gegen sie geprüft.

| Checkpoint | p(` per`) | Rang | Top-1 |
|---|---|---|---|
| 130000 | 0.09402 | 3 | `ph` 0.726 |
| 131000 | 0.09087 | **2** | `ph` 0.798 |
| 132000 | 0.10461 | 3 | `ph` 0.744 |
| 133000 | 0.10833 | 3 | `ph` 0.638 |

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
# voller Lauf
python -m m60482.run --passages /pfad/zu/passagen.json --output runs

# ohne GPU und ohne Modell: Proben gegen ein Bundle mit den echten Ankerzahlen
python -m m60482.run --smoke

# nur bestimmte Proben, auf einem bereits gemessenen Bundle
python -m m60482.run --reuse-bundle runs/<id>/bundle.npz --probes noise_floor context_dependence
```

### Tests

```bash
pytest tests/ -q      # 29 Tests, keine GPU, kein Download, kein Netz
```

---

## Aufbau

```
m60482/
  config.py            eingefrorene Fakten + Referenzwerte der früheren Läufe
  passages.py          laden, verifizieren, vier Herkunftsstufen
  model.py             Checkpoint-Laden, fp32, eager attention, Determinismus
  measure.py           Bundle: alles, was ein Forward-Pass je Passage liefert
  aux.py               Trunkierungsleiter, Greedy-Fortsetzung, Namensersetzung, untrainiert
  stats.py             kontrollbezogene p-Werte und die Grenzen, die darin stehen
  registry.py          der Proben-Vertrag
  papers.py            die Kandidaten und woher jeder kam
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

### Reihenfolge

Die billigen Proben laufen zuerst, weil sie die Prämisse kippen können. Ist das Zieltoken nie
die Greedy-Fortsetzung, beantworten die teuren mechanistischen Proben eine Frage, die niemand
stellen sollte.

| # | Probe | Frage |
|---|---|---|
| 10 | `memorization_entry` | Ist das Ziel je die Greedy-Fortsetzung? |
| 15 | `noise_floor` | Überschreitet die Änderung das Checkpoint-Rauschen? |
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

**fp32, nicht fp16.** Bei step131000 liegen Ziel (0.09087) und Konkurrent `"/"` (0.08965)
0.00122 auseinander, und die Schlagzeilen-Statistik ist ein *Rang*. In fp16 ist diese Ordnung
nicht zuverlässig reproduzierbar. fp32-Gewichte sind ~5.7 GB; eine A100-40GB hat reichlich Platz.

**`attn_implementation="eager"`.** Unter SDPA und FlashAttention gibt `output_attentions=True`
stillschweigend `None` zurück. Dann fehlen die Attention-Zahlen, statt falsch zu sein — was
schlimmer ist, weil es keinen Fehler auslöst. `model.forward_pass` prüft das und wirft.

---

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

- Mit 40 Kontrollen ist der kleinste erreichbare p-Wert **1/41 = 0.0244**. Nach Korrektur für
  die Zahl der Proben ist hier nichts signifikant. Das ist eine Eigenschaft des Designs, nicht
  der Daten.
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
- `global_sample_index` ist suite-spezifisch. `pythia-1.4b` und `pythia-1.4b-deduped` haben
  verschiedene Datenreihenfolgen, also bezeichnet Index 134428942 in beiden eine andere Sequenz.
  Welche Suite den Anker erzeugt hat, entscheidet die Architektur nicht — beide Configs sind
  byteidentisch.
- Für Pythia-**1.4B** existiert öffentlich **kein** SAE (nur 70m, 160m, 410m, 1b, 2.8b), und
  die vorhandenen sind für den finalen Checkpoint trainiert. Die Methode des Repeat-Curse-Papers
  ist hier also nicht ausführbar; ein Final-Checkpoint-SAE auf ein step131000-Modell anzuwenden
  würde ein anderes Objekt messen.
