# Metodologia badania

## Problem i zakres

Badanie dotyczy klasyfikacji etykiet jakości powietrza w jednym publicznym zbiorze
tabelarycznym. Nie jest prognozą przyszłego stanu powietrza: plik nie zawiera czasu.
Nie jest też modelem indywidualnej rekomendacji zdrowotnej.

Główne zadanie zachowuje cztery oryginalne klasy. Zadanie dodatkowe porównuje dwa
sposoby otrzymywania tej samej informacji binarnej.

| Etykieta źródłowa | Kod czteroklasowy | Kod binarny |
|---|---:|---:|
| Good | 0 | 0 |
| Moderate | 1 | 0 |
| Poor | 2 | 1 |
| Hazardous | 3 | 1 |

Wariant binarny ma klasę dodatnią `Poor/Hazardous`. To odwrotna orientacja względem
historycznej zmiennej `Can_Go_Outside`, w której 1 oznaczało `Good/Moderate`.
Nieznane lub brakujące etykiety powodują błąd zamiast domyślnego przypisania do klasy.

## Pytania badawcze i kryteria sukcesu

1. Czy klasyfikacja czterech oryginalnych etykiet przewyższa przewidywanie klasy
   najczęstszej przy poprawnie oddzielonym treningu i ocenie?
2. Czy na wspólnym celu binarnym bezpośredni model różni się od sumy
   `P(Poor) + P(Hazardous)` modelu czteroklasowego?
3. Jak zmienia się jakość regresji logistycznej po zastosowaniu wag klas lub SMOTE?
4. Jaki sygnał niosą zanieczyszczenia i cechy kontekstu? Jakie znaczenie mają CO,
   skorelowane frakcje PM, czyszczenie i odległości do klastrów?

Główna metryka: macro-F1. Każda klasa ma jednakową wagę, mimo różnicy liczebności.
Dodatkowo raportowane są accuracy, balanced accuracy, precision/recall/F1 klas,
log-loss oraz opis błędów o większej odległości na skali uporządkowanych kategorii.
Nie przyjęto arbitralnego wymogu osiągnięcia 98% accuracy. Kryteriami ukończenia są
odtwarzalność, brak dopasowywania transformacji do walidacji/testu i jawne ograniczenia.

## Pochodzenie danych

Źródło: Mujtaba Mateen, *Air Quality and Pollution Assessment*, Kaggle, wersja 2.
Metadane autora zapisano w `data/source_metadata.json`. Licencja wskazana przez
Kaggle: Apache 2.0. Oryginalny plik użytkownika zachowano bez zmian bajtowych.

Opis autora nie podaje miast, stacji, dat, procedury pomiaru ani reguły tworzenia
etykiet. Nie potwierdza źródeł WHO/World Bank i nie określa jednoznacznie, czy dane
są rzeczywiste czy syntetyczne. Takich twierdzeń nie przyjmujemy jako faktów.
Nie dodajemy danych z innych źródeł o niezgodnych jednostkach lub etykietach.

## Audyt i czyszczenie

Audyt całego pliku obejmuje schemat, wartości skończone, braki, duplikaty, rozkłady
etykiet oraz z góry ustalone reguły domenowe. Nie dopasowuje transformacji do testu.
EDA rozkładów i korelacji korzysta z development.

Główny wariant przetwarzania:

- wilgotność poza [0, 100] i ujemne stężenia, odległość lub gęstość zaludnienia
  zastępuje brakami;
- mediany do imputacji wyznacza w aktualnej części treningowej;
- zachowuje obserwacje z dużymi dodatnimi stężeniami i poprawne ujemne temperatury;
- nie usuwa całych rekordów;
- relację PM2.5 > PM10 tylko flaguje z uwagi na brak danych o porównywalności pomiarów.

Ujemne wskazania mogą w rzeczywistych systemach wynikać z korekcji instrumentu.
Brak odpowiedniej dokumentacji jest ograniczeniem przyjętej reguły, dlatego
porównujemy ją z zachowaniem wartości surowych. Reguły nie są zmieniane po teście.

Duplikaty cech są odrzucane z komunikatem o potrzebie podziału grupowego, aby identyczne
obserwacje nie mogły trafić do obu części losowego podziału. Pipeline przyjmuje braki
części cech, ale odrzuca rekord całkowicie pozbawiony obserwowanych wartości,
wartości nieskończone i tekst w kolumnach numerycznych.

## Podział i zagnieżdżona walidacja

Konfiguracja `configs/study.json` została ustalona przed głównym przebiegiem.
Nie jest to zewnętrzna prerejestracja badania.

1. 80% development i 20% test, `random_state=42`, stratyfikacja według czterech klas.
2. W obu zadaniach te same rekordy oraz te same zewnętrzne foldy: pięć foldów
   stratyfikowanych względem czterech klas, tasowanie, `random_state=19`.
3. Dla każdej rodziny w każdym zewnętrznym foldzie osiem losowanych konfiguracji,
   ocenianych wewnętrzną trzykrotną walidacją stratyfikowaną względem danego celu.
4. Każda próba dopasowuje pipeline od początku. SMOTE nie jest wykonywany na
   zbiorze przed przekazaniem go do walidacji.
5. Rodzinę wybiera średnie zewnętrzne macro-F1. Następnie jej hiperparametry są
   ponownie wybierane na całym development tym samym wewnętrznym wyszukiwaniem.
6. Przed testem zapisywane są modele, hiperparametry i progi w `frozen_decisions.json`.
7. Następuje jedna końcowa faza oceny ustalonych procedur. Wszystkie wyniki testowe
   są raportowane, również gdy nie potwierdzają oczekiwań.

Wybór najwyższego wyniku spośród rodzin dodaje optymizm do średniej zewnętrznej
najlepszej rodziny. Nie przedstawiamy jej jako bezstronnej oceny całego procesu
wyboru rodziny. Wydzielony test służy odrębnej ocenie tej decyzji.

Starszy projekt analizował już ten sam publiczny zbiór. Nowy test jest proceduralnym
holdoutem poprawionej analizy, nie prospektywnym zbiorem zewnętrznym. Nie ma podstaw
do twierdzeń o skuteczności w innych miastach, stacjach i okresach.

## Modele i przestrzenie wyszukiwania

Wszystkie metody korzystają z imputacji i skalowania. Dla drzew liniowe skalowanie
nie dodaje nowej informacji, ale umożliwia wspólną, prawidłową geometrię SMOTE.
Do wyszukiwania trafia brak równoważenia, wagi klas lub SMOTE. kNN nie ma wag klas.
SMOTE korzysta z trzech sąsiadów, co pozwala także na mały test techniczny.

| Rodzina | Strojone parametry | Konfiguracje w pełnej przestrzeni |
|---|---|---:|
| Regresja logistyczna | C: 0.1, 1, 10; równoważenie | 9 |
| kNN | k: 7, 15, 31; uniform/distance; bez SMOTE/z SMOTE | 12 |
| SVM RBF | C: 0.5, 2, 8; gamma: scale/0.1; równoważenie | 18 |
| Random Forest | min_samples_leaf: 1, 3, 7; max_features: sqrt/0.8; równoważenie | 18 |
| Histogram Gradient Boosting | learning_rate: 0.05/0.12; max_leaf_nodes: 7/15; L2: 0/1; równoważenie | 24 |

W każdej przestrzeni losowanych jest osiem konfiguracji, bez dostosowywania budżetu
do wyniku testowego. Równy budżet liczby konfiguracji nie oznacza równego czasu CPU.
Random Forest ma 160 drzew; boosting 150 iteracji bez early stopping.

SVM korzysta z prawdopodobieństw dostarczanych przez SVC. Ponieważ standardowe
`SVC.predict` może różnić się od `argmax(predict_proba)`, `ProbabilitySVC` jawnie
stosuje drugą regułę w treningowej selekcji, ocenie i inferencji. Odpowiada ona
progowi 0.5 w zadaniu binarnym. Kalibracja SVC jest wewnętrzną częścią dopasowania;
nie stanowi niezależnej walidacji prawdopodobieństw, zwłaszcza po resamplingu.
Jej jakość sprawdzają log-loss, Brier i wykres kalibracji na końcowym teście.

## Eksperymenty wrażliwości

Porównania wykorzystują te same zewnętrzne podziały development i stałe modele
referencyjne. Nie uruchamiają wtórnego strojenia na teście.

- Równoważenie: regresja logistyczna z C=1, bez równoważenia / z wagami / z SMOTE.
- Cechy: ustalony Histogram Gradient Boosting, wszystkie / tylko zanieczyszczenia /
  tylko kontekst / bez CO / bez PM2.5.
- Klastry: ten sam boosting z dodatkowymi dwiema odległościami do centroidów KMeans
  zbudowanego na skalowanych CO i odległości od przemysłu. Centroidy są dopasowywane
  tylko do treningu danego foldu. Nie tworzymy ułamkowych etykiet klastra przez SMOTE.
- Czyszczenie: ten sam boosting, wartości surowe kontra maskowanie i imputacja.
- Krzywa uczenia: konfiguracja wybrana dla czterech klas, cztery wielkości próby,
  te same zewnętrzne foldy; diagnostyka, a nie nowy wybór hiperparametrów.

Wyniki ablation są zależne od modelu referencyjnego i opisowe. Nie wykonujemy serii
testów istotności na zależnych wynikach foldów.

## Binaryzacja, progi i niepewność

Podstawowe porównanie binarne obejmuje dwie ustalone procedury, obie z progiem 0.5:
bezpośredni model i agregację prawdopodobieństw czterech klas. Nie porównujemy
bezpośrednio macro-F1 dwóch i czterech klas jako miary przewagi jednego zadania.

Dodatkowe progi maksymalizują F2 na zagnieżdżonych predykcjach out-of-fold wybranej
rodziny z development. F2 daje większą wagę recall. Jest preferencją modelowania,
nie oszacowaniem kosztów zdrowotnych. Siatka: 0.05–0.95 co 0.01. Remis rozstrzyga
odległość od 0.5. Wartość F2 wykorzystana do doboru progu jest optymistyczna.

95% przedziały percentylowe wyznaczamy przez 2000 resamplowań testu z powtórzeniami
wewnątrz czterech oryginalnych klas, przy stałych modelach. Podana niepewność jest
warunkowa względem proporcji klas i nie obejmuje treningu ani zmiany domeny.
Różnica macro-F1 modeli binarnych korzysta z tych samych indeksów bootstrapowych.
Bootstrap i test sparowanej poprawności zakładają w przybliżeniu niezależne rekordy.
Brak metadanych nie pozwala sprawdzić ani uwzględnić zależności wewnątrz stacji i okresów.
Dokładny test McNemara dotyczy sparowanej poprawności, nie hipotezy równości F1.
Przedział różnicy zawierający zero nie dowodzi równoważności metod.

## Interpretacja i ślad odtwarzania

Permutation importance dotyczy zamrożonego modelu czteroklasowego i końcowego testu.
Nie służy do ponownego doboru cech. Skorelowane cechy mogą dzielić lub ukrywać
znaczenie. PDP dla z góry wybranych CO i odległości od przemysłu uśredniamy po
development; nie interpretujemy go przyczynowo.

Manifest obejmuje hash danych, konfigurację, wersje głównych bibliotek, hashe modułów
obliczeniowych i bazowy commit checkoutu. Modele są zapisywane lokalnie. W repozytorium
są predykcje testowe, OOF, wyniki każdego foldu, wyniki wewnętrznego strojenia wybranych
rodzin, audyt, raport i wykonany notebook. To pozwala zweryfikować tabele niezależnie
od zapisanych modeli. Znaczniki czasu i czasy wykonania nie są deterministyczne.

## Odniesienia

- [Kaggle: opis i metadane zbioru](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment).
- [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- [imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html).
- [scikit-learn: Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002).
  [SMOTE](https://doi.org/10.1613/jair.953). JAIR, 16, 321–357.
- Varma, S., Simon, R. (2006).
  [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91).
  BMC Bioinformatics, 7, 91.
