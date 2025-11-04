import os
import csv
import pickle


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg  # Utilisation de scipy.linalg pour la décomposition en valeurs propres généralisées
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from pipeline import *  # Import des fonctions corrigées
from pathlib import Path
import numpy as np
import cv2, os

# Import necessary PyQt5 components
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QCheckBox, QGroupBox, QRadioButton, QLineEdit, QSlider, QProgressBar, 
    QComboBox, QTabWidget, QPlainTextEdit, QTableWidget, QSplitter, QAction, 
    QToolBar, QFileDialog, QMessageBox, QHeaderView, QTableWidgetItem, QPushButton,
    QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import utility libraries
import sys
import csv
from io import BytesIO

# Conditional imports for export functionalities
try:
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import letter
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_OK = True
except ImportError:
    PPTX_OK = False

try:
    from prettytable import PrettyTable
    PRETTYTABLE_OK = True
except ImportError:
    PRETTYTABLE_OK = False