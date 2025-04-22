import numpy as np
from scipy.optimize import minimize
from scipy.stats import triang
import pandas as pd

# === PARAMETRY OGÓLNE ===
np.random.seed(15)
n_przebiegi = 300

miesiace = ["czerwiec", "lipiec", "sierpien"]
n_miesiace = len(miesiace)

# Parametry popytu
mean_popyt = [1106, 1802, 1977]
std_popyt = [100, 150, 200]

# Czas produkcji (min): Triangular(15, 20, 30)
czas_prod_rozk = triang(c=(20-15)/(30-15), loc=15, scale=15)

# Opóźnienie płatności (dni): Triangular(10, 30, 90)
platnosc_opoznienie_rozk = triang(c=(30-10)/(90-10), loc=10, scale=80)

# Parametry kosztowe
cena_jednostkowa = 125
materialy_na_jednostke = 2.0
cena_materialu_kg = 3.53
godz_na_jednostke = [0.3030, 0.1918, 0.3154]
maks_godz_dzien = 8 * 22  # 8h * 22 dni robocze
stawka_plac = 20
stawka_nadgodzin = 22.5
min_placa_gwarant = 6750

zmienna_stopa_kszt = 10
stale_kszt= 10000
amortyzacja_kszt = 1000

zm_stp_kszt_admin = 5
stale_kszt_admin = 5000
amortyzacja_admin = 500

zakup_sprzetu = [70000, 30000, 0]
utrzymywanie_kredytu = [20000, 20000, 20000]
dywidendy = [0, 0, 10000]

mies_st_proc = 0.0975
pocz_gotowka = 10000
wymagana_gotowka = 10000
pocz_zapasy = 221
pozadane_zapasy_konc = [360, 395, 269]
pocz_zapasy_kg = 250
pozadane_zapasy_konc_kg = [367, 370, 242]

# === FUNKCJA SYMULUJĄCA JEDEN SCENARIUSZ ===
def simulate_scenario(pracownicy_na_miesiac):
    gotowka = pocz_gotowka
    zapasy_jednostki = pocz_zapasy
    zapasy_kg = pocz_zapasy_kg
    kredyt = 0
    TOTAL_CASHFLOWS = []

    for mies_idx in range(n_miesiace):
        # --- Losowanie popytu, czasu produkcji, opóźnienia ---
        popyt = np.random.normal(loc=mean_popyt[mies_idx], scale=std_popyt[mies_idx]) #popyt losujemy z rozkladu normalnego o parametrach
        czas_prod = czas_prod_rozk.rvs() 
        platnosc_opoznienie = platnosc_opoznienie_rozk.rvs()

        pracownicy = pracownicy_na_miesiac[mies_idx]
        godz_wymagane_na_jednostke = godz_na_jednostke[mies_idx]

        # --- Ustalenie wielkości produkcji ---
        minuty_na_dzien = 8*60  # 8h dziennie
        wymagane_jednostki = popyt + pozadane_zapasy_konc[mies_idx] - zapasy_jednostki
        moce_prod = pracownicy*minuty_na_dzien/czas_prod
        wyprod_jednostki = min(moce_prod, max(0, wymagane_jednostki))

        # --- Zużycie materiałów ---
        wymag_materialy = wyprod_jednostki * materialy_na_jednostke
        CALKOWITE_kg = wymag_materialy + pozadane_zapasy_konc_kg[mies_idx] - zapasy_kg
        zakupione_kg = max(0, CALKOWITE_kg)

        # --- Koszty materiałów ---
        koszt_materialow = zakupione_kg * cena_materialu_kg

        # --- Czas pracy i płace ---
        CALKOWITE_GODZ = wyprod_jednostki * godz_wymagane_na_jednostke
        regularne_godz = min(pracownicy * maks_godz_dzien, CALKOWITE_GODZ)
        nadgodziny = max(0, CALKOWITE_GODZ - regularne_godz)
        koszt_pracy = max(regularne_godz * stawka_plac + nadgodziny * stawka_nadgodzin, min_placa_gwarant)

        # --- Koszty ogólne i administracyjne ---
        KOSZTY_ZMIENNE = wyprod_jednostki * zmienna_stopa_kszt
        KOSZTY_CALKOWITE = KOSZTY_ZMIENNE + stale_kszt+ amortyzacja_kszt
        koszty_zmienne_admin = popyt * zm_stp_kszt_admin
        koszty_admin_CALKOWITE = koszty_zmienne_admin + stale_kszt_admin + amortyzacja_admin

        # --- Sprzedaż i wpływy gotówki ---
        sprzedaz = min(popyt, wyprod_jednostki + zapasy_jednostki)
        przychod = sprzedaz * cena_jednostkowa
        zdyskontowany_przychod = przychod / ((1 + mies_st_proc) ** (platnosc_opoznienie / 30))

        # --- Gotówkowe wydatki ---
        gotowka_odplywy = (
                    0.5*koszt_materialow +
                    koszt_pracy +
                    KOSZTY_CALKOWITE +       
                    koszty_admin_CALKOWITE +          
                    zakup_sprzetu[mies_idx] +
                    utrzymywanie_kredytu[mies_idx] +
                    dywidendy[mies_idx]
                )

        # --- Saldo gotówki ---
        gotowka += zdyskontowany_przychod
        gotowka -= gotowka_odplywy

        # --- Obsługa kredytu ---
        if gotowka < wymagana_gotowka:
            potrzebny_kredyt = wymagana_gotowka - gotowka
            stopa_proc = potrzebny_kredyt * mies_st_proc
            kredyt += potrzebny_kredyt + stopa_proc
            gotowka += potrzebny_kredyt
        else:
            # Spłata kredytu jeśli gotówka pozwala
            if kredyt > 0:
                platnosc = min(gotowka - wymagana_gotowka, kredyt)
                kredyt -= platnosc
                gotowka -= platnosc

        TOTAL_CASHFLOWS.append(gotowka)

        # --- Przeniesienie zapasów na kolejny miesiąc ---
        zapasy_jednostki = pozadane_zapasy_konc[mies_idx]
        zapasy_kg = pozadane_zapasy_konc_kg[mies_idx]

    return TOTAL_CASHFLOWS[-1]  # gotowka na koniec sierpnia

# === FUNKCJA CELU: OCZEKIWANY gotowka NA KONIEC SIERPNIA ===
def oczekiw_gotowka(x):
    CALKOWITE = 0
    for _ in range(n_przebiegi):
        CALKOWITE += simulate_scenario(x)
    return -CALKOWITE / n_przebiegi  # Maksymalizujemy => dajemy minus

# === OPTYMALIZACJA ===
def optimise():
    bounds = [(5, 50)] * n_miesiace  # Zakres pracowników na miesiąc
    punkt_poczatkowy = [100, 100, 100]
    punkt_poczatkowy2 = [1000, 1000, 1000] #!!!!! SPRAWDŹ -> MAKSIMUM TROCHĘ INNE => BŁĄD PRÓBKOWANIA => ZA MAŁE n_przebiegi!!!
    #punkt_poczatkowy = [7,7,7]
    punkt_poczatkowy3 = [1, 1, 1]
    wynik = minimize(oczekiw_gotowka, punkt_poczatkowy, bounds=bounds, method="SLSQP")
    wynik2 = minimize(oczekiw_gotowka, punkt_poczatkowy2, bounds=bounds, method="SLSQP")
    wynik3 = minimize(oczekiw_gotowka, punkt_poczatkowy3, bounds=bounds, method="SLSQP")
    optimum_liczby_pracownikow = wynik.x
    wart_oczekiw_gotowka = -wynik.fun

    # print(" Optymalna liczba pracowników na miesiąc dla pkt. początkowego [100,100,100]:")
    # for i, m in enumerate(miesiace):
    #     print(f"  {m.capitalize()}: {optimum_liczby_pracownikow[i]:.2f} pracowników")

    # print(f" Oczekiwana gotówka na koniec sierpnia: ${wart_oczekiw_gotowka:,.2f}")

    # optimum_liczby_pracownikow2 = wynik2.x
    # wart_oczekiw_gotowka2 = -wynik2.fun
    # print(" Optymalna liczba pracowników na miesiąc dla pkt poczatkowego [1000,1000,1000]:")
    # for i, m in enumerate(miesiace):
    #     print(f"  {m.capitalize()}: {optimum_liczby_pracownikow2[i]:.2f} pracowników")
    # print(f" Oczekiwana gotówka na koniec sierpnia: ${wart_oczekiw_gotowka2:,.2f}")


    # optimum_liczby_pracownikow3 = wynik3.x
    # wart_oczekiw_gotowka3 = -wynik3.fun
    # print(" Optymalna liczba pracowników na miesiąc dla pkt poczatkowego [1,1,1]:")
    # for i, m in enumerate(miesiace):
    #     print(f"  {m.capitalize()}: {optimum_liczby_pracownikow3[i]:.2f} pracowników")

    # print(f" Oczekiwana gotówka na koniec sierpnia: ${wart_oczekiw_gotowka3:,.2f}")

    return wart_oczekiw_gotowka

optimise()

#====== ANALIZA WRAZLIWOSCI 

"""Dla zadanej zmiennej przeprowadza analize wrazliwosci, zmieniajac jej wartosc w zakresie +- 10% od wartosci bazowej. Zwraca listę wartości gotówki na koniec sierpnia dla każdej zmiennej.
Do celów analizy wraliwości przyjmuję jeden punkt poczatkowy [100,100,100] dla liczby pracowników."""

def analiza_wrazliwosci(zmienna, wartosc_bazowa):
    wartosci = []
    for zmienna in range(int(0.5*wartosc_bazowa) , int(1.5*wartosc_bazowa) + 1 , 1):#zmiana co 1 jednostkę w zakresie +- 50% - ta funkcja działa tylko dla zmiennych całkowitych!
        # Przeprowadzamy optymalizację dla zmiennej
        wart_oczekiw_gotowka = optimise()
        # Store the result (e.g., expected cash flow)
        wartosci.append({
            'zmienna': zmienna,
            'wartosc_gotowka': float(wart_oczekiw_gotowka)  # Oczekiwana gotówka na koniec sierpnia
        })
    return wartosci

def analiza_wrazliwosci_triangle(variable_name, min_value, mode_value, max_value, num_samples=300):
    """
   Analiza wrazliwosci dla zmiennych trojkatnych.

    Args:
        variable_name (str): Name of the variable to analyze.
        min_value (float): Minimum value of the triangular distribution.
        mode_value (float): Most likely value (mode) of the triangular distribution.
        max_value (float): Maximum value of the triangular distribution.
        num_samples (int): Number of samples to draw from the triangular distribution.

    Returns:
        pd.DataFrame: Results of the sensitivity analysis.
    """
    # Calculate the triangular distribution parameters
    c = (mode_value - min_value) / (max_value - min_value)
    triangular_dist = triang(c, loc=min_value, scale=(max_value - min_value))

    # Sample values from the triangular distribution
    samples = triangular_dist.rvs(size=num_samples)

    # Perform sensitivity analysis
    results = []
    for sample in samples:
        # Run the simulation 
        result = optimise()  # Call the optimise function to get the cash flow

        # Store the result
        results.append({
            'sample_value': sample,
            'cash_flow': result
        })

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results