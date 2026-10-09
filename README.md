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
und ohne Störungsstatus sind und an denen beide Fahrtrichtungen gezählt haben.

## Fahrplan

1. [ ] Daten laden, zu Tageswerten aggregieren und explorativ analysieren (Notebook: Wochenmuster, Saison, Ferien, Ausreißer)
2. [ ] Features: Wetter (Open-Meteo), Wochentag, Feiertage NRW, Semesterzeiten, Lags
3. [ ] Modelle: naive Baseline, lineare Regression, LightGBM mit zeitlicher Kreuzvalidierung
4. [ ] Streamlit-App mit der Prognose für morgen
