# Dane i atrybucja

- Autor: **Mujtaba Mateen**, identyfikator Kaggle `mujtabamatin`.
- Zbiór: [Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment).
- Wersja źródła w chwili weryfikacji: **2**, aktualizacja **2024-12-04**.
- Licencja wskazana w metadanych Kaggle: **Apache 2.0**; tekst w [LICENSE](../LICENSE).
- Metadane autora: [source_metadata.json](source_metadata.json), pobrane przez publiczne API Kaggle.
- Plik użytkownika: [raw/pollution_dataset.csv](raw/pollution_dataset.csv), zachowany bajtowo.
- SHA-256: `052fc6679ac05ef51e9198cb80b7e944ea134a04ad07b86dd53688e0f86895d6`.

Licencja i atrybucja źródłowego zbioru pozostają w mocy. Dane w `raw/` nie są
modyfikowane przez eksperymenty. Czyszczenie i tworzenie celu są odtwarzalne w kodzie.

| Kolumna | Jednostka według autora |
|---|---|
| Temperature | °C |
| Humidity | % wilgotności względnej |
| PM2.5 | µg/m³ |
| PM10 | µg/m³ |
| NO2 | ppb |
| SO2 | ppb |
| CO | ppm |
| Proximity_to_Industrial_Areas | km do najbliższej strefy przemysłowej |
| Population_Density | osoby/km² |
| Air Quality | Good, Moderate, Poor, Hazardous |

Opis nie dostarcza nazw miast, identyfikatorów stacji, dat, definicji okresu
uśredniania, procedury pobierania danych ani algorytmu przypisywania etykiet.
Nie pozwala potwierdzić źródeł WHO/World Bank ani pomiarowego lub syntetycznego
charakteru obserwacji. Projekt nie dopisuje takiego pochodzenia.

Dokumentacja autora nie wystarcza do utożsamienia klas z określonym oficjalnym
indeksem AQI. Jednostek nie przeliczamy na inne standardy bez informacji o pomiarach.
