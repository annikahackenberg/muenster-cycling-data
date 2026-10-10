# muenster-cycling-data

Exploring Münster's bicycle traffic with open data – analysis, forecasting and an AI assistant.

Erstes Projekt: **Tagesprognose der Radfahrten an der Zählstelle Warendorfer Straße** (Daten seit Januar 2019).

## Daten

Quelle: [od-ms/radverkehr-zaehlstellen](https://github.com/od-ms/radverkehr-zaehlstellen), Stadt Münster,
Lizenz [Datenlizenz Deutschland Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0).
Die Stadt veröffentlicht 15-Minuten-Zählungen pro Zählstelle als eine CSV-Datei je Monat. Es sind Rohdaten:
Ausfälle und Baustellen sind nicht bereinigt.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Tageswerte erzeugen

```bash
python -m radmuenster
```

Lädt alle Monatsdateien nach `data/raw/` (wird gecacht) und schreibt die Tagessummen nach
`data/processed/daily_100034983.parquet`. Die Spalte `complete` markiert Tage, die vollständig
und ohne Störungsstatus sind und an denen beide Fahrtrichtungen plausibel gezählt haben.

## Erkenntnisse aus der explorativen Analyse

Details, Plots und Herleitung: [notebooks/01_explorative_analyse.ipynb](notebooks/01_explorative_analyse.ipynb)
(Datenstand 09.10.2026).

**Datenqualität**
- Der Statuswert der Stadt reicht nicht, um Ausfälle zu erkennen. Stadteinwärts war zweimal rund
  80 Tage lang leer (Herbst 2021 und Herbst 2023), dazu kommen mehrere Teilausfälle, alle mit Status „in Ordnung“.
  Erkannt werden sie über den Anteil der beiden Fahrtrichtungen (`direction_fault`).
- Von April 2021 bis Mai 2025 fehlt in den Daten der letzte Tag jedes Monats.
- Der Anteil stadteinwärts springt zweimal dauerhaft, jeweils direkt nach einer Störung (Juni 2024, Oktober 2025).
  Das deutet auf Wartung oder Änderungen an der Zählstelle hin. Werte vor und nach einem Sprung sind
  nicht voll vergleichbar.

**Muster**
- **Wochentag** ist der stärkste Effekt: Sonntags fahren rund 40 % so viele wie an einem Werktag, mittwochs
  und samstags (Markttage) am meisten.
- **Saison** ist schwächer (Januar und Dezember etwa 13 % unter, Mai bis November etwa 3–8 % über dem
  Jahresmittel), mit einer Delle im April (Ostern, vorlesungsfreie Zeit).
- **Ausreißer** sind fast alle erklärbar: Feiertage, Weihnachten, Rosenmontag, Schnee im Februar 2021.
  Der Rest sind vor allem Sonntage mit besonders gutem oder schlechtem Wetter.
- **Autokorrelation** ist bei 7 Tagen am höchsten (0,70) und bei 364 Tagen hoch (0,58). Lag 1 steht für eine
  Prognose für morgen nicht zur Verfügung (Leakage), nutzbar sind Lags ab 2 Tagen.

## Fahrplan

1. [x] Daten laden, zu Tageswerten aggregieren und explorativ analysieren (Wochenmuster, Saison, Autokorrelation, Ausreißer, Datenqualität)
2. [ ] Features: Wetter (Open-Meteo), Wochentag, Feiertage NRW, Semesterzeiten, Lags
3. [ ] Modelle: naive Baseline, lineare Regression, LightGBM mit zeitlicher Kreuzvalidierung
4. [ ] Streamlit-App mit der Prognose für morgen
