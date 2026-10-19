# Archiwum projektu źródłowego

`original_project.md` to niezmieniona kopia opracowania dostarczonego przez autora
projektu. Zachowuje historię analizy, w tym błędy i niezweryfikowane twierdzenia.
Nie stanowi aktualnej instrukcji ani źródła finalnych wyników.

Poprawiona metodologia jest w `docs/methodology.md`, a aktualne wyniki w
`reports/report.html` oraz `reports/holdout_metrics.json`.

Najważniejsze poprawki obejmują użycie przekazanych argumentów zamiast globalnych
danych, fold-local preprocessing i resampling, skalowanie metod odległościowych,
wybór modeli bez wyników testowych, zachowanie czterech klas i kontrolę jakości
fizycznych wartości. Historyczna funkcja celu sugerująca możliwość wyjścia na
zewnątrz została zastąpiona jawną klasyfikacją etykiet zbioru.
