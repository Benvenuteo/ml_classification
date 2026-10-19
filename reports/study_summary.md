# Wyniki i wnioski

- Cztery klasy: testowe macro-F1 = 0.9426. Nie ma empirycznej konieczności redukowania celu do dwóch klas w tym zbiorze.

- Recall klasy Hazardous wynosi 86.0% przy 100 obserwacjach testowych tej klasy. Wynik globalny nie zastępuje analizy pomyłek dotyczących najrzadszej kategorii.

- Równoważenie w regresji logistycznej: bez równoważenia 0.9108, SMOTE 0.9127 (różnica +0.0019), wagi klas 0.9107 (różnica -0.0001). To średnie development, zależne od modelu referencyjnego.

- Model referencyjny z samym kontekstem uzyskuje 0.8084, z samymi zanieczyszczeniami 0.8552, ze wszystkimi cechami 0.9332. Usunięcie CO zmienia wynik o -0.0396. Sygnał kontekstu wymaga ostrożności wobec nieudokumentowanego sposobu tworzenia etykiet.

- Maskowanie wartości zamiast zachowania surowych zmienia macro-F1 modelu referencyjnego o +0.0013. Dodanie odległości do klastrów zmienia je o -0.0024. Małe różnice opisowe nie stanowią samodzielnego dowodu istotnej poprawy.

- Przedział różnicy binarnego macro-F1 obejmuje zero. Ten test nie rozstrzyga przewagi modelu bezpośredniego lub agregacji i nie dowodzi ich równoważności.

Szczegóły: report.html, holdout_metrics.json oraz tables/. Wyniki dotyczą klasyfikacji etykiet w tym zbiorze, bez walidacji zewnętrznej.
