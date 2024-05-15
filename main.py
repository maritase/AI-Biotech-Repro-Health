import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QAction, QColorDialog, QMenuBar)
from PyQt5.QtGui import QPixmap, QIcon, QColor, QImage
from PyQt5.QtCore import Qt

class SpermAnalyzer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sperm Analyzer")
        self.setWindowIcon(QIcon("icon.png"))  # Remplacez "icon.png" par votre fichier d'icône

        self.init_ui()

    def init_ui(self):
        # Créer les éléments de l'interface utilisateur
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid black;")  # Bordure noire autour de l'image

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)

        self.results_label = QLabel("Results:")

        self.field_count_label = QLabel("Fields: 0/5")
        self.sperm_count_label = QLabel("Spermatozoide: 0/500")

        # Créer un menu
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")
        color_menu = menu_bar.addMenu("Color")

        # Actions du menu
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Changer la couleur de fond de l'image
        change_image_color_action = QAction("Change Image Color", self)
        change_image_color_action.triggered.connect(self.change_image_color)
        color_menu.addAction(change_image_color_action)

        # Disposition
        main_layout = QVBoxLayout()
        main_layout.addWidget(menu_bar)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.image_label)

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.load_button)
        bottom_layout.addWidget(self.analyze_button)
        bottom_layout.addWidget(self.results_label)
        bottom_layout.addWidget(self.field_count_label)
        bottom_layout.addWidget(self.sperm_count_label)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap)

    def analyze(self):
        if self.image_label.pixmap():
            # Convertir le QPixmap en numpy array
            qimg = self.image_label.pixmap().toImage()
            qimg = qimg.convertToFormat(4)  # Convertir en format RGBA
            width = qimg.width()
            height = qimg.height()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # Convertir en numpy array
            
            # Convertir l'image en niveaux de gris
            gray_image = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            
            # Appliquer un seuillage pour détecter les contours
            _, thresh = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)
            
            # Trouver les contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Dessiner les contours sur l'image originale
            sperm_count = len(contours)
            result_image = cv2.drawContours(arr, contours, -1, (0, 255, 0), 2)
            
            # Mettre à jour les étiquettes de résultats
            self.field_count_label.setText(f"Fields: 1/1")  # Pour cette démonstration, nous supposons un seul champ d'analyse
            self.sperm_count_label.setText(f"Spermatozoa: {sperm_count}/500")  # Supposons un maximum de 500 spermatozoïdes
            
            # Afficher l'image mise à jour
            height, width, channel = result_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(result_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.image_label.setPixmap(pixmap)
        else:
            print("Please load an image first.")

    def change_image_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.image_label.setStyleSheet(f"border: 2px solid black; background-color: {color.name()};")
    


    def process_image(self, file_name):
        # Charger l'image
        image = cv2.imread(file_name)

        # Charger le modèle YOLO
        net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
        layers = net.getLayerNames()
        output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Prétraiter l'image
        blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Interpréter les résultats
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Dessiner la boîte englobante
                    box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (x, y, w, h) = box.astype("int")
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # Dessiner l'étiquette de la classe
                    label = f"Class: {class_id}"
                    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Afficher l'image
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpermAnalyzer()
    window.show()
    sys.exit(app.exec_())
