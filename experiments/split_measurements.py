import pandas as pd
import os

# Florian Liszt: Masterarbeit
# output_{Nummer}_{Setup}_{Klasse}.csv
# --- KONFIGURATION ---
num_scenarios = 19       # Anzahl der Szenarien (1 bis 19)
setups = ['A']      # Welche Setups existieren?
labels = ['c', 'w'] # c = correct, w = wrong, i = inversed

# Pfad zu dem Ordner, in dem deine kombinierten Dateien liegen.
# (Achte darauf, ob der Ordner MASTER oder MASTERS heißt!)
base_folder = 'raw_data/MASTER_THESIS_DATASET'

# Erstelle die Ziel-Ordner (Test_Bay_F1 und F2) IN dem Dataset-Ordner, falls sie noch nicht existieren
for bay in ['Test_Bay_F1', 'Test_Bay_F2']:
    os.makedirs(os.path.join(base_folder, bay, 'Extracted_Measurements'), exist_ok=True)

# Schleife über alle Szenarien-Nummern (1 bis num_scenarios)
for i in range(1, num_scenarios + 1):
    for setup in setups:
        for label in labels:
            # Generiere den Dateinamen, z.B. "output_4_A_c.csv"
            file_name = f"output_{i}_{setup}_{label}.csv"
            
            # Der vollständige Pfad zur kombinierten Datei
            source_path = os.path.join(base_folder, file_name)
            
            # Prüfe, ob die Datei in dem Ordner liegt
            if not os.path.exists(source_path):
                continue # Datei existiert nicht -> ohne Fehlermeldung überspringen
                
            print(f"Verarbeite {file_name}...")
            
            try:
                # Lade die kombinierte CSV-Datei (mit Semikolon, wie in deiner Excel)
                df = pd.read_csv(source_path, sep=';') 

                # 1. Filtere Spalten für F2 (Ortsnetztrafo)
                cols_f2 = [col for col in df.columns if 'ortsnetztrafo' in col or col == 'Timestamp']
                df_f2 = df[cols_f2]
                f2_path = os.path.join(base_folder, 'Test_Bay_F2', 'Extracted_Measurements', file_name)
                
                # WICHTIG: Speichere im alten Format (Trenner = Komma, Dezimal = Komma)
                df_f2.to_csv(f2_path, sep=',', decimal=',', index=False)

                # 2. Filtere Spalten für F1 (Smart Meter)
                cols_f1 = [col for col in df.columns if 'smartmeter' in col or col == 'Timestamp']
                df_f1 = df[cols_f1]
                f1_path = os.path.join(base_folder, 'Test_Bay_F1', 'Extracted_Measurements', file_name)
                
                # WICHTIG: Speichere im alten Format (Trenner = Komma, Dezimal = Komma)
                df_f1.to_csv(f1_path, sep=',', decimal=',', index=False)
                
            except Exception as e:
                print(f"Fehler beim Verarbeiten von {file_name}: {e}")

print("Aufteilung und Format-Anpassung aller Szenarien abgeschlossen!")