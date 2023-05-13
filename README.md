# SI_Projekt
Kod składa się z pięciu części:

0. Wykorzystywane biblioteki
1. Tworzenie modelu sieci neuronowej. Definiowane są warstwy konwolucyjne, które wykrywają wzorce w danych wejściowych. Następnie są definiowane warstwy w pełnie połączone, które przetwarzają wyniki z warstw konwolucyjnych, aby uzyskać ostateczne przewidywania.
2. Przygotowanie danych. Dane treningowe i testowe są ładowane z katalogów, gdzie znajdują się obrazy. Transformacje są stosowane do obu zestawów danych, takie jak zmiana obrazów, zmiana na tensor i normalizacja.
3. Tworzenie obiektów typu DataLoader, który umożliwia ładowanie danych do modelu w porcjach z ustawieniem losowego przemieszania danych. Są tworzone dwa obiekty DataLoader dla danych treningowych i testowych.
4. Trenowanie i testowanie sieci neuronowej. Sieć neuronowa jest trenowana za pomocą algorytmu propagacji wstecznej, który minimalizuje funkcję straty, czyli Cross Entropy Loss. Optymalizator SGD jest używany do aktualizowania wag sieci. Model jest trenowany przez 50 epok, a po każdej epoce wyświetlane są wyniki straty. Następnie sieć jest testowana na danych testowych, a dokładność klasyfikacji jest wyświetlana na ekranie.

Uruchomienie programu:
Kod można uruchomić w Pycharm w pliku main.py. 
Linie kodu 86-90 są zakomentowane. Pozwalają one wyświetlić testowane obrazy i sprawdzić czy przewidywania programu były poprawne. 
0 - oznacza linie ciągłą
1 - oznacza pasy dla pieszych
2 - oznacza linie przerywaną

Predicted oznacza przewidywaną odpowiedź kodu. Actual oznacza jaka powinna być odpowiedź.
