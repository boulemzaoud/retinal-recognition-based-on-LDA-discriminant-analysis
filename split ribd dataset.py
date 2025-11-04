import os
import shutil
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, QLineEdit, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
import sys

class RIBDDatasetSplitterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_path = ""
        self.output_dir = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('RIBD Dataset Splitter (Logique Configurable)')
        self.setGeometry(100, 100, 550, 400) # Augmenté la hauteur pour les nouveaux champs

        main_layout = QVBoxLayout()

        # Section pour la sélection du dossier du Dataset
        self.dataset_label = QLabel("Aucun dossier de dataset sélectionné.")
        main_layout.addWidget(self.dataset_label)

        self.dataset_button = QPushButton("Sélectionner le dossier du Dataset RIBD")
        self.dataset_button.clicked.connect(self.select_dataset_folder)
        main_layout.addWidget(self.dataset_button)

        # Section pour la sélection du dossier de sortie
        self.output_label = QLabel("Aucun dossier de sortie sélectionné.")
        main_layout.addWidget(self.output_label)

        self.output_button = QPushButton("Sélectionner le dossier de sortie")
        self.output_button.clicked.connect(self.select_output_folder)
        main_layout.addWidget(self.output_button)
        
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Section pour les paramètres de division (Train/Test X values)
        params_group_label = QLabel("Paramètres de division (valeurs 'X' de IM0000X_Y.JPG):")
        params_group_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(params_group_label)

        # Layout pour les valeurs X de l'entraînement
        train_x_layout = QHBoxLayout()
        self.train_x_label = QLabel("Valeurs X pour l'entraînement (ex: 1,2,3,4):")
        train_x_layout.addWidget(self.train_x_label)
        self.train_x_input = QLineEdit("1,2,3,4") # Valeur par défaut
        self.train_x_input.setToolTip("Entrez les indices X des images pour l'ensemble d'entraînement, séparés par des virgules.")
        train_x_layout.addWidget(self.train_x_input)
        main_layout.addLayout(train_x_layout)

        # Layout pour les valeurs X du test
        test_x_layout = QHBoxLayout()
        self.test_x_label = QLabel("Valeurs X pour le test (ex: 5):")
        test_x_layout.addWidget(self.test_x_label)
        self.test_x_input = QLineEdit("5") # Valeur par défaut
        self.test_x_input.setToolTip("Entrez les indices X des images pour l'ensemble de test, séparés par des virgules.")
        test_x_layout.addWidget(self.test_x_input)
        main_layout.addLayout(test_x_layout)
        
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Bouton pour lancer la division
        self.split_button = QPushButton("Diviser le Dataset")
        self.split_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; font-size: 14px;} QPushButton:hover { background-color: #45a049; }")
        self.split_button.clicked.connect(self.run_split_dataset_process)
        main_layout.addWidget(self.split_button)

        # Label pour les messages de statut/résultat
        self.status_label = QLabel("") 
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        main_layout.addStretch(1) # Ajoute un espace flexible en bas

        self.setLayout(main_layout)

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier du Dataset RIBD")
        if folder:
            self.dataset_path = folder
            self.dataset_label.setText(f"Dossier Dataset : {Path(self.dataset_path).name}")
            self.dataset_label.setToolTip(self.dataset_path)
            print(f"Dossier Dataset sélectionné : {self.dataset_path}")
            self.status_label.setText("")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if folder:
            self.output_dir = folder
            self.output_label.setText(f"Dossier de sortie : {Path(self.output_dir).name}")
            self.output_label.setToolTip(self.output_dir)
            print(f"Dossier de sortie sélectionné : {self.output_dir}")
            self.status_label.setText("")

    def _parse_x_values(self, x_values_str, field_name):
        """Helper function to parse comma-separated integer strings."""
        if not x_values_str.strip():
            # Permettre des listes vides si l'utilisateur ne veut pas de fichiers pour un ensemble
            return [] 
        try:
            values = [int(x.strip()) for x in x_values_str.split(',')]
            # Vérifier les doublons
            if len(values) != len(set(values)):
                QMessageBox.warning(self, "Valeurs dupliquées", f"Les valeurs X pour '{field_name}' contiennent des doublons. Veuillez vérifier.")
                return None # Indique une erreur
            return values
        except ValueError:
            QMessageBox.critical(self, "Erreur de saisie", 
                                 f"Les valeurs X pour '{field_name}' ne sont pas valides. "
                                 f"Elles doivent être des nombres entiers séparés par des virgules (ex: 1,2,3). "
                                 f"Reçu : '{x_values_str}'")
            return None # Indique une erreur

    def run_split_dataset_process(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Entrée manquante", "Veuillez sélectionner le dossier du dataset RIBD.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Sortie manquante", "Veuillez sélectionner le dossier de sortie.")
            return

        train_x_str = self.train_x_input.text()
        test_x_str = self.test_x_input.text()

        train_x_values = self._parse_x_values(train_x_str, "l'entraînement")
        if train_x_values is None: # Erreur de parsing
            return

        test_x_values = self._parse_x_values(test_x_str, "le test")
        if test_x_values is None: # Erreur de parsing
            return
        
        # Vérification de la non-superposition des listes train et test
        common_values = set(train_x_values) & set(test_x_values)
        if common_values:
            QMessageBox.warning(self, "Conflit de valeurs X", 
                                f"Les valeurs X suivantes sont présentes à la fois dans l'entraînement et le test : {common_values}. "
                                "Veuillez vous assurer que les ensembles sont disjoints.")
            return

        if not train_x_values and not test_x_values:
             QMessageBox.warning(self, "Aucune valeur X spécifiée", 
                                "Veuillez spécifier des valeurs X pour l'entraînement et/ou le test.")
             return


        try:
            self.status_label.setText("Traitement en cours...")
            QApplication.processEvents() # Pour s'assurer que le label se met à jour

            result_message = split_ribd_dataset(self.dataset_path, self.output_dir, train_x_values, test_x_values)
            
            QMessageBox.information(self, "Succès", "La division du dataset est terminée avec succès !")
            self.status_label.setText(result_message)
            print(result_message)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Erreur de Fichier", f"Fichier ou dossier non trouvé : {e}")
            self.status_label.setText(f"Erreur : {e}")
        except PermissionError as e:
            QMessageBox.critical(self, "Erreur de Permission", f"Permission refusée : {e}")
            self.status_label.setText(f"Erreur : {e}")
        except ValueError as e: # Capturer les erreurs de valeur de split_ribd_dataset (si photo_id mal formé)
            QMessageBox.critical(self, "Erreur de Valeur", f"Erreur lors de l'analyse des noms de fichiers : {e}")
            self.status_label.setText(f"Erreur : {e}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur Inattendue", f"Une erreur s'est produite lors de la division : {e}")
            self.status_label.setText(f"Erreur : {e}")
            print(f"Erreur détaillée : {type(e).__name__} - {e}")


def split_ribd_dataset(dataset_path_str, output_dir_str, train_x_values, test_x_values):
    """
    Divise le dataset RIBD en ensembles d'entraînement et de test basés sur les valeurs X configurées.
    - Pour chaque personne Y, on garde les images IM0000X_Y.JPG
    - Les images avec X dans train_x_values vont dans le train
    - Les images avec X dans test_x_values vont dans le test
    """

    dataset_path = Path(dataset_path_str)
    output_dir = Path(output_dir_str)

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Le dossier du dataset spécifié n'existe pas : {dataset_path}")

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    
    # Nettoyer les dossiers de sortie s'ils existent pour éviter les accumulations
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
        
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    image_dict = {}  # Regroupement par person_id (Y)
    # Chercher .JPG et .jpg pour plus de flexibilité
    image_files = list(dataset_path.glob("*.JPG")) + list(dataset_path.glob("*.jpg"))

    for img_file in sorted(image_files):
        name = img_file.stem  # ex. IM000003_1
        parts = name.split("_")
        
        if len(parts) != 2:
            print(f"Skipping file with unexpected name format: {img_file.name}")
            continue
        
        photo_id_str, person_id_str = parts  # photo_id_str est "IM0000X", person_id_str est "Y"

        try:
            # Extrait la partie numérique X de photo_id_str (ex: "IM00001" -> 1)
            # Gère les préfixes comme "IM" ou "image" suivis de chiffres.
            numeric_part_of_photo_id = ""
            for char in reversed(photo_id_str):
                if char.isdigit():
                    numeric_part_of_photo_id = char + numeric_part_of_photo_id
                else:
                    break # Arrêter dès qu'on ne trouve plus de chiffre en partant de la fin
            
            if not numeric_part_of_photo_id: # Si aucune partie numérique n'a été trouvée
                 print(f"Skipping file, cannot parse photo index from: {photo_id_str} in {img_file.name}")
                 continue
            
            x_val = int(numeric_part_of_photo_id)

        except ValueError:
            print(f"Skipping file, cannot parse photo index from: {photo_id_str} in {img_file.name}")
            continue

        if person_id_str not in image_dict:
            image_dict[person_id_str] = []
        image_dict[person_id_str].append((x_val, img_file))  # stocker X (int) et le chemin du fichier

    # Split pour chaque personne
    train_count = 0
    test_count = 0
    skipped_due_to_x_value = 0

    for person_id, files_with_x in image_dict.items():
        # Pas besoin de trier ici si on ne sélectionne que par appartenance à train_x_values ou test_x_values
        # sorted_files_with_x = sorted(files_with_x, key=lambda item: item[0]) 

        for x_val, img_f in files_with_x: # On itère sur les fichiers tels qu'ils ont été lus
            copied = False
            if x_val in train_x_values:
                try:
                    shutil.copy(img_f, train_dir / img_f.name)
                    train_count += 1
                    copied = True
                except Exception as e:
                    print(f"Error copying {img_f.name} to train for person {person_id}: {e}")
            
            # Utiliser 'elif' si un fichier ne peut appartenir qu'à un seul ensemble.
            # Si un fichier peut appartenir aux deux (ce qui est évité par la vérification de non-superposition),
            # alors utiliser 'if' séparés. La vérification de non-superposition rend 'elif' plus logique.
            elif x_val in test_x_values:
                try:
                    shutil.copy(img_f, test_dir / img_f.name)
                    test_count += 1
                    copied = True
                except Exception as e:
                    print(f"Error copying {img_f.name} to test for person {person_id}: {e}")
            
            if not copied:
                # Ce fichier n'a pas été assigné ni à train ni à test basé sur les X_values fournis.
                print(f"Info: Image {img_f.name} (X={x_val}) for person {person_id} not assigned to train or test based on provided X values.")
                skipped_due_to_x_value +=1


    result_message = (f"✅ Division terminée.\n"
                      f"Entraînement : {train_count} images.\n"
                      f"Test : {test_count} images.\n")
    if skipped_due_to_x_value > 0:
        result_message += f"{skipped_due_to_x_value} images non assignées (valeur X non spécifiée pour train/test)."
        
    return result_message

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RIBDDatasetSplitterApp()
    window.show()
    sys.exit(app.exec_())
