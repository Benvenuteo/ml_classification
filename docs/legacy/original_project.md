# Cel Projektu

Celem projektu jest zastosowanie wybranych metod klasyfikacyjnych do predykcji zmiennej binarnej, która sugeruje możliwość wyjścia na zewnątrz w zależności od różnych czynników.<br> W projekcie wykorzystano zbiór danych (5000) zawierający informacje m.in. o poziomie zanieczyszczenia powietrza w różnych miastach na świecie.<br> Dane pochodzą ze strony kaggle.com (https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment).<br> Są wynikiem web scrapingu ze źródeł takich jak World Health Organization (WHO) (https://www.who.int/health-topics/air-pollution) oraz World Bank Data (https://data.worldbank.org/indicator/EN.POP.DNST).<br>

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import dalex as dx
from sklearn.inspection import PartialDependenceDisplay
from IPython.display import Image
from IPython.display import SVG
```

# EDA - eksploracyjna analiza danych

```python
df = pd.read_csv('pollution_dataset.csv')
df.head()
```

Output:

|  | Temperature | Humidity | PM2.5 | PM10 | NO2 | SO2 | CO | Proximity\_to\_Industrial\_Areas | Population\_Density | Air Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 29.8 | 59.1 | 5.2 | 17.9 | 18.9 | 9.2 | 1.72 | 6.3 | 319 | Moderate |
| 1 | 28.3 | 75.6 | 2.3 | 12.2 | 30.8 | 9.7 | 1.64 | 6.0 | 611 | Moderate |
| 2 | 23.1 | 74.7 | 26.7 | 33.8 | 24.4 | 12.6 | 1.63 | 5.2 | 619 | Moderate |
| 3 | 27.1 | 39.1 | 6.1 | 6.3 | 13.5 | 5.3 | 1.15 | 11.1 | 551 | Good |
| 4 | 26.5 | 70.7 | 6.9 | 16.0 | 21.9 | 5.6 | 1.01 | 12.7 | 303 | Good |

Zbiór danych zawiera następujące kolumny:
- Temperature: temperatura (°C),
- Humidity: wilgotność względna (%),
- PM2.5: poziom drobnych cząstek stałych (µg/m³),
- PM10: poziom cząstek stałych (µg/m³),
- NO2: poziom dwutlenku azotu (ppb),
- SO2: poziom dwutlenku siarki (ppb),
- CO: poziom tlenku węgla (ppm),
- Proximity_To_Industrial_Areas: odległość od najbliższej strefy przemysłowej (km),
- Population_Density: gęstość zaludnienia (osób/km²),
- Air Quality: ocena jakości powietrza, przymuje wartości:<br>
&nbsp;- Good: dobra,<br>
&nbsp;- Moderate: umiarkowana,<br>
&nbsp;- Poor: zła,<br>
&nbsp;- Hazardous: niebezpieczna.<br>

```python
df.describe()
```

Output:

|  | Temperature | Humidity | PM2.5 | PM10 | NO2 | SO2 | CO | Proximity\_to\_Industrial\_Areas | Population\_Density |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 5000.000000 | 5000.000000 | 5000.000000 | 5000.000000 | 5000.000000 | 5000.000000 | 5000.000000 | 5000.000000 | 5000.000000 |
| mean | 30.029020 | 70.056120 | 20.142140 | 30.218360 | 26.412100 | 10.014820 | 1.500354 | 8.425400 | 497.423800 |
| std | 6.720661 | 15.863577 | 24.554546 | 27.349199 | 8.895356 | 6.750303 | 0.546027 | 3.610944 | 152.754084 |
| min | 13.400000 | 36.000000 | 0.000000 | -0.200000 | 7.400000 | -6.200000 | 0.650000 | 2.500000 | 188.000000 |
| 25% | 25.100000 | 58.300000 | 4.600000 | 12.300000 | 20.100000 | 5.100000 | 1.030000 | 5.400000 | 381.000000 |
| 50% | 29.000000 | 69.800000 | 12.000000 | 21.700000 | 25.300000 | 8.000000 | 1.410000 | 7.900000 | 494.000000 |
| 75% | 34.000000 | 80.300000 | 26.100000 | 38.100000 | 31.900000 | 13.725000 | 1.840000 | 11.100000 | 600.000000 |
| max | 58.600000 | 128.100000 | 295.000000 | 315.800000 | 64.900000 | 44.900000 | 3.720000 | 25.800000 | 957.000000 |

Średnia temperatura w zbiorze danych wynosi 30°C, a wilgotność względna 70%. Sugeruje to pochodzenie rekordów z obszarów o ciepłym i wilgotnym klimacie. Taki typ klimatu jest spotykany w regionach tropikalnych lub subtropikalnych.<br> Wysokie temperatury w połączeniu z wysoką wilgotnością mogą wpływać na jakość powietrza, co z kolei może mieć wpływ na zmienną objaśnianą.

```python
df.info()
```

Output:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 10 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   Temperature                    5000 non-null   float64
 1   Humidity                       5000 non-null   float64
 2   PM2.5                          5000 non-null   float64
 3   PM10                           5000 non-null   float64
 4   NO2                            5000 non-null   float64
 5   SO2                            5000 non-null   float64
 6   CO                             5000 non-null   float64
 7   Proximity_to_Industrial_Areas  5000 non-null   float64
 8   Population_Density             5000 non-null   int64  
 9   Air Quality                    5000 non-null   object 
dtypes: float64(8), int64(1), object(1)
memory usage: 390.8+ KB
```

Dane nie zawierają braków.<br>
Wszystkie zmienne, poza 'Air Quality', są typu liczbowego.<br>
W celu utworzenia zmiennej objaśnianej, która będzie sugerowała możliwość wyjścia na zewnątrz, zmienna 'Air Quality' została przekształcona w zmienną 'Can_Go_Outside' według następujących kryteriów:<br>
- 0 - nie można wyjść na zewnątrz, jeśli jakość powietrza (Air Quality) jest zła (Poor) lub niebezpieczna (Hazardous)
- 1 - można wyjść na zewnątrz, jeśli jakość powietrza (Air Quality) jest dobra (Good) lub umiarkowana (Moderate).<br>

W ten sposób otrzymano zmienną binarną odpowiednią do zastosowania w modelach klasyfikacyjnych.

```python
df['Can_Go_Outside'] = df['Air Quality'].apply(lambda x: 1 if x in ['Moderate', 'Good'] else 0)
df = df.drop('Air Quality', axis=1)
```

```python
num_cols = 2
num_rows = (len(df.columns) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, column in enumerate(df.columns):
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column].hist(ax=axes[i], bins=40)
        axes[i].set_title(f'Rozkład zmiennej {column}')
    else:
        df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[i])
        axes[i].set_title(f'Rozkład zmiennej {column}')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```

Output:

```
<Figure size 1500x2500 with 10 Axes>
```

Możemy zaobserwować dwa wyraźne piki w rozkładzie zmiennych 'Proximity_to_Industrial_Areas' i 'CO'.<br>
Świadczy to o bimodalnym rozkładzie tych zmiennych. Mogą one mieć dwie różne subpopulacje lub grupy w zbiorze danych.<br>
W celu analizy danych o takim charakterze warto rozważyć klasteryzację.<br>
Jest to technika, która polega na grupowaniu obiektów w taki sposób, aby obiekty w tej samej grupie (klastrze) były bardziej podobne do siebie niż do obiektów w innych grupach.<br>
Dodanie zmiennej 'Cluster' ma na celu przypisanie każdemu rekordowi w zbiorze danych etykiety klastra, do którego należy.<br>
Dzięki temu można analizować i porównywać różne grupy danych, co może pomóc w identyfikacji wzorców i zależności.

```python
variables = ['Proximity_to_Industrial_Areas', 'CO']

kmeans = KMeans(n_clusters=2, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[variables])

for variable in variables:
    plt.figure(figsize=(12, 6))
    for cluster in df['Cluster'].unique():
        subset = df[df['Cluster'] == cluster]
        subset[variable].hist(alpha=0.5, label=f'Cluster {cluster}', bins=40)
    plt.title(f'Histogram zmiennej {variable} z podziałem na klastry')
    plt.legend()
    plt.show()
```

Output:

```
<Figure size 1200x600 with 1 Axes>
```

```
<Figure size 1200x600 with 1 Axes>
```

```python
# Calculate correlations
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

Output:

```
<Figure size 1200x800 with 2 Axes>
```

Z macierzy korrelacji wynika, że zmienne 'PM2.5' i 'PM10' są ze sobą silnie skorelowane (0.89).<br>
Wynika to z faktu, iż występowanie cząsteczek stałych PM2.5 oraz PM10 jest związane z podobnymi źródłami emisji, takimi jak np. spalanie paliw kopalnych.<br>
Zmienne objaśniające powinny być wybierane w taki sposób, aby były silnie skorelowane ze zmienną objaśnianą a równocześnie słabo skorelowane między sobą.<br>
W związku z tym, w przypadku przeuczenia modelu (overfitting), warto rozważyć usunięcie jednej z tych zmiennych.

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature'], df['CO'], alpha=0.5)
plt.title('Związek pomiędzy temperaturą powietrza a występowaniem cząstek tlenku węgla')
plt.xlabel('Temperatura (Temperature)')
plt.ylabel('Tlenek węgla (CO)')
plt.grid(True)
plt.show()
```

Output:

```
<Figure size 1000x600 with 1 Axes>
```

Korelacja pomiędzy temperaturą powietrza a poziomem tlenku węgla ma charakter dodatni, ale jest umiarkowana i nieliniowa.<br>
W wyższych temperaturach, szczególnie latem, często występują stabilne warunki atmosferyczne, takie jak brak wiatru i inwersja termiczna.<br>
Inwersja powoduje, że zanieczyszczenia, w tym tlenek węgla, są uwięzione blisko powierzchni ziemi, co prowadzi do ich kumulacji.<br>
To zjawisko znajduje odwzorowanie w powyższym wykresie.

# Podział danych na zbiór uczący i testowy

Podział na zbiór uczący i testowy pozwala na ocenę jakości klasyfikacji modelu.<br>
Zbiór uczący jest wykorzystywany do trenowania modelu, natomiast zbiór testowy do oceny jego jakości.<br>
W celu zminimalizowania ryzyka przeuczenia modelu, dane podzielono na zbiory uczący i testowy w proporcji 80:20.<br>

```python
X = df.drop('Can_Go_Outside', axis=1)
y = df['Can_Go_Outside']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
df['Can_Go_Outside'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
plt.title('Rozkład zmiennej objaśnianej (Can_Go_Outside)')
plt.ylabel('')
plt.show()
```

Output:

```
<Figure size 640x480 with 1 Axes>
```

Zbiór danych jest delikatnie niezbalansowany.<br>
W celu jego zrównoważenia zastosowano jedną z technik oversamplingu, SMOTE.<br>
Synthetic Minority Oversampling Technique polega na generowaniu sztucznych rekordów dla mniejszej klasy, które są podobne do istniejących rekordów tej klasy.<br>

```python
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("Oryginalny rozkład zmiennej:", y_train.value_counts())
print("Rozkład zmiennej po zbalansowaniu:", y_train_balanced.value_counts())
```

Output:

```
Oryginalny rozkład zmiennej: Can_Go_Outside
1    2797
0    1203
Name: count, dtype: int64
Rozkład zmiennej po zbalansowaniu: Can_Go_Outside
0    2797
1    2797
Name: count, dtype: int64
```

Do klasyfikacji zastosowano następujące metody uczenia maszynowego:<br>
- regresję logistyczną,
- metodę k-najbliższych sąsiadów,
- maszynę wektorów nośnych,
- las losowy,
- oraz wzmocnienie gradientowe.<br>

W celu oceny jakości klasyfikacji zastosowano miarę dokładności (accuracy) oraz raport klasyfikacji.<br>
Raport klasyfikacji zawiera miary takie jak: precyzja, czułość, F1-score oraz wskaźnik podobieństwa Jaccarda.<br>
Najlepszy model klasyfikacji to taki, który osiąga najwyższą wartość miary dokładności oraz inne miary klasyfikacji na poziomie zbliżonym do 1.<br>
Spośród wszystkich modeli wybrano dwa, które osiągnęły najwyższe wartości, a następnie zastosowano technikę Grid Search w celu znalezienia najlepszych hiperparametrów dla obu modeli.<br>

```python
def machine_learning_analysis(X_train_balanced, X_test, y_train_balanced, y_test ):

    models = {
        'Regresja logistyczna': LogisticRegression(max_iter=2000),
        'K-najbliższych sąsiadów': KNeighborsClassifier(),
        'Maszyna wektorów nośnych': SVC(probability=True),
        'Las losowy': RandomForestClassifier(random_state=42),
        'Boosting gradientowy': GradientBoostingClassifier(random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            'Dokładność': accuracy,
            'Predykcje': y_pred
        }
        print(f"{model_name} - Dokładność: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

results = {}

machine_learning_analysis(X_train_balanced, X_test, y_train_balanced, y_test )
```

Output:

```
Regresja logistyczna - Dokładność: 0.97
              precision    recall  f1-score   support

           0       0.95      0.95      0.95       297
           1       0.98      0.98      0.98       703

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000

K-najbliższych sąsiadów - Dokładność: 0.91
              precision    recall  f1-score   support

           0       0.88      0.80      0.84       297
           1       0.92      0.96      0.94       703

    accuracy                           0.91      1000
   macro avg       0.90      0.88      0.89      1000
weighted avg       0.91      0.91      0.91      1000

Maszyna wektorów nośnych - Dokładność: 0.90
              precision    recall  f1-score   support

           0       0.88      0.78      0.83       297
           1       0.91      0.96      0.93       703

    accuracy                           0.90      1000
   macro avg       0.90      0.87      0.88      1000
weighted avg       0.90      0.90      0.90      1000

Las losowy - Dokładność: 0.98
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       297
           1       0.99      0.99      0.99       703

    accuracy                           0.98      1000
   macro avg       0.98      0.98      0.98      1000
weighted avg       0.98      0.98      0.98      1000

Boosting gradientowy - Dokładność: 0.98
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       297
           1       0.99      0.99      0.99       703

    accuracy                           0.98      1000
   macro avg       0.98      0.98      0.98      1000
weighted avg       0.98      0.98      0.98      1000
```

Najlepsze wyniki klasyfikacji uzyskano dla modelu Random Forest oraz Gradient Boosting.<br>
Oba modele osiągnęły dokładność na poziomie 0.98.<br>
Zobaczmy, czy ten wynik może ulec poprawie.<br>
W celu poprawy wyniku zastosujemy technikę Grid Search, która pozwala na przeszukanie przestrzeni hiperparametrów w celu znalezienia najlepszych parametrów modelu.<br>
Algorytm lasu losowego posiada następujące hiperparametry:<br>
- n_estimators: liczba drzew w lesie,<br>
- max_depth: maksymalna głębokość drzewa,<br>
- min_samples_split: minimalna liczba próbek wymagana do podziału węzła wewnętrznego,<br>
- min_samples_leaf: minimalna liczba próbek wymagana w węźle liścia.<br>

Gradient Boosting zaś:<br>
- n_estimators: liczba drzew w lesie,<br>
- learning_rate: współczynnik uczenia,<br>
- max_depth: maksymalna głębokość drzewa,<br>
- min_samples_split: minimalna liczba próbek wymagana do podziału węzła wewnętrznego,<br>

Przyjrzyjmy się najlepszym parametrom oraz wynikom uzyskanym dla obu modeli:

```python
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_balanced, y_train_balanced)

grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train_balanced, y_train_balanced)

print("Najlepsze parametry dla lasu losowego:", grid_search_rf.best_params_)
print("Uzyskany wynik:", grid_search_rf.best_score_)
print("Najlepsze parametry dla modelu ze wzmocnieniem gradientowym:", grid_search_gb.best_params_)
print("Uzyskany wynik:", grid_search_gb.best_score_)
```

Output:

```
Fitting 5 folds for each of 108 candidates, totalling 540 fits
Fitting 5 folds for each of 243 candidates, totalling 1215 fits
Najlepsze parametry dla lasu losowego: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Uzyskany wynik: 0.9819456101393879
Najlepsze parametry dla modelu ze wzmocnieniem gradientowym: {'learning_rate': 0.2, 'max_depth': 4, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}
Uzyskany wynik: 0.9830187955320445
```

Nieznacznie lepszy wynik uzyskano dla modelu wzmocnienia gradientowego.<br>
Zobaczmy jakie zmienne mają największy wpływ na predykcje modelu.

```python
best_params_gb = grid_search_gb.best_params_
gb = GradientBoostingClassifier(**best_params_gb, random_state=42)
gb.fit(X_train_balanced, y_train_balanced)

importances = gb.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': X_train_balanced.columns,
    'Importance': importances
})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print(feature_importances)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Znaczenie cech dla modelu wzmocnienia gradientowego')
plt.show()
```

Output:

```
                         Feature    Importance
6                             CO  7.844846e-01
7  Proximity_to_Industrial_Areas  9.023214e-02
4                            NO2  4.722238e-02
5                            SO2  2.257061e-02
1                       Humidity  2.207162e-02
0                    Temperature  1.834540e-02
8             Population_Density  9.593676e-03
3                           PM10  4.551848e-03
2                          PM2.5  9.268570e-04
9                        Cluster  8.508752e-07
```

```
<Figure size 1200x600 with 1 Axes>
```

Największy wpływ na predykcje modelu ma zmienna określająca poziom tlenku węgla (CO) z ważnością na poziomie 0.78.<br>
Drugą najistotniejszą zmienną okazała się odległość od strefy przemysłowej (Proximity_to_Industrial_Areas) 0.1,<br>
Warto zauważyć, że zmienna 'Cluster' nie ma wpływu na predykcje modelu.<br>
Została ona utworzona w celu analizy danych o bimodalnym rozkładzie.<br>
W przypadku, gdy zmienna nie wpływa na predykcje modelu, warto rozważyć jej usunięcie.<br>

Interpretacja profili Ceteris Paribus polega na analizie, jak zmieniają się predykcje modelu, gdy zmienia się wartość jednej cechy, przy założeniu, że wszystkie inne cechy pozostają stałe.

```python
Image(filename='newplot.png')
```

Output:

```
<IPython.core.display.Image object>
```

Zobaczmy wykresy częściowej zależności dla wybranych zmiennych: 'Temperature', 'CO' oraz 'Proximity_to_Industrial_Areas' .

```python
PartialDependenceDisplay.from_estimator(estimator=gb, X=X_train_balanced, features=[0, 6, 7], grid_resolution=50)
```

Output:

```
<sklearn.inspection._plot.partial_dependence.PartialDependenceDisplay at 0x25986ef7e60>
```

```
<Figure size 640x480 with 4 Axes>
```

Zależność temperatury jest negatywna.<br>
Wykres pokazuje, że wzrost temperatury obniża wynik, co sugeruje, że wysoka temperatura może mieć negatywny wpływ na przewidywaną zmienną.<br>
Duże zmiany wartości przewidywanej przy małym wzroście CO mogą sugerują, że zmienna ta jest bardzo ważną cechą dla modelu.<br>
Wzrost wartości Proximity_to_Industrial_Area stabilizuje się dla wyższych wartości cechy, co sugeruje, że po pewnym progu dodatkowa bliskość do obszarów przemysłowych nie wpływa znacząco na wynik.

# Podsumowanie

*Spośród wszystkich modeli klasyfikacji najlepsze wyniki uzyskano dla modelu wzmocnienia gradientowego.<br><br>
Model osiągnął dokładność na poziomie 0.98. Wynik jest zadowalający i świadczy o wysokiej skuteczności w predykcji zmiennej objaśnianej.<br><br>
W celu poprawy wyniku zastosowano technikę oversamplingu SMOTE oraz przeszukano przestrzeń hiperparametrów za pomocą Grid Search, jednak nie wpłynęło to istotnie na poprawę wyniku.<br><br>
Najważniejsze zmienne dla modelu to: CO, Proximity_to_Industrial_Areas oraz NO2.<br><br>
Dodatkowo zaobserwowano, że po pewnym progu dodatkowa bliskość do obszarów przemysłowych nie wpływa znacząco na wynik.*