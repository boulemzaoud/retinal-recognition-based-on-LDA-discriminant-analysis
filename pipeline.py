"""
PyQt5 – Reconnaissance biométrique rétinienne
LBP multiscale + LDA(lsqr, shrinkage)     
+ TFA/TFR/TEE • Courbe DET • Histogramme LBP      •  Export CSV / PDF / PPTX
© 2025
"""
# ----------------------------------------------------------------------------- 
# Imports
# -----------------------------------------------------------------------------
import os, sys, time, csv, pickle, warnings
import cv2, numpy as np, pandas as pd
from skimage.morphology import skeletonize
from skimage.feature    import local_binary_pattern
from skimage.measure    import label, regionprops
from skimage.filters    import frangi
from scipy.interpolate  import interp1d
from typing import Tuple
from models import CustomLDA



from sklearn.pipeline             import make_pipeline
from sklearn.preprocessing        import StandardScaler, label_binarize

from sklearn.metrics              import (classification_report, confusion_matrix,
                                          roc_curve, auc, RocCurveDisplay,
                                         )
from sklearn.model_selection      import cross_val_score
from sklearn.exceptions           import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ------------------------------------------------------------------ #
# Compatibilité scikit-image : grey*/gray*                           #
# ------------------------------------------------------------------ #
try:                                # ➜ versions récentes (>= 0.19)
    from skimage.feature import greycomatrix, greycoprops
except ImportError:                 # ➜ versions anciennes  (<= 0.18)
    from skimage.feature import graycomatrix as greycomatrix
    from skimage.feature import graycoprops   as greycoprops


from skimage.feature import hog
from sklearn.decomposition import PCA




import matplotlib
matplotlib.use("Agg")     # sécurité hors écran
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from prettytable import PrettyTable

try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_OK = True
except ImportError:
    PPTX_OK = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdf_canvas
    PDF_OK = True
except ImportError:
    PDF_OK = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
    QMessageBox, QProgressBar, QRadioButton, QButtonGroup, QLineEdit, QPlainTextEdit,
    QComboBox, QCheckBox, QGroupBox, QToolBar, QAction, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QSizePolicy, QHeaderView, QSlider
)
from PyQt5.QtGui  import QPixmap, QImage, qRgb, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# ----------------------------------------------------------------------------- 
# Utils : conversions & pré-/post-traitements
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------#
# 1.  Utilitaires image
# -----------------------------------------------------------------------------#
from PyQt5.QtGui import QImage, QPixmap, qRgb
import cv2, numpy as np

def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """
    Convertit un ndarray OpenCV/Numpy en QPixmap.
    • (H,W)         : N&B 8-bit
    • (H,W,3) BGR   : couleur
    • (H,W,4) BGRA  : couleur + alpha
    """
    if img is None:
        raise ValueError("Image is None")

    if img.ndim == 2:                                    # -------- grayscale
        h, w = img.shape
        qimg = QImage(img.data, w, h,
                      img.strides[0],
                      QImage.Format_Indexed8)
        qimg.setColorTable([qRgb(i, i, i) for i in range(256)])

    elif img.ndim == 3:                                  # -------- couleur
        h, w, c = img.shape
        if c == 3:                                       # BGR ➜ RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, w, h,
                          img_rgb.strides[0],
                          QImage.Format_RGB888)
        elif c == 4:                                     # BGRA ➜ RGBA
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            qimg = QImage(img_rgba.data, w, h,
                          img_rgba.strides[0],
                          QImage.Format_RGBA8888)
        else:
            raise ValueError(f"{c} canaux non gérés (attendu 3 ou 4).")

    else:
        raise ValueError(f"Shape inattendue : {img.shape}. "
                         "Attendu (H,W) ou (H,W,3/4).")

    return QPixmap.fromImage(qimg)


def clahe(img):      return cv2.createCLAHE(1.5, (8, 8)).apply(img)
def denoise(img):    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
def vessels(img):    return (frangi(img.astype(np.float32)/255.)*255).astype(np.uint8)
def morph_close(b):  return cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
def preprocess(img, sz=(256,256)): return cv2.equalizeHist(cv2.resize(img, sz))
def binarize(img):   return cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

try:
    from cv2.ximgproc import thinning as cv_thin
    USE_THIN = True
except Exception:
    USE_THIN = False
def skeleton(b):     return cv_thin(b) if USE_THIN else skeletonize(b>0).astype(np.uint8)*255

# -----------------------------------------------------------------------------#
# 2.  Extraction de caractéristiques
# -----------------------------------------------------------------------------#
# 2-1  LBP multiscale
def lbp_hist_multiscale(img, radii=(1,2,3)):
    feats=[]
    for r in radii:
        h,_ = np.histogram(
            local_binary_pattern(img, 8*r, r, 'uniform').ravel(),
            bins=np.arange(0,8*r+3), density=True
        )
        feats.extend(h)
    return np.asarray(feats, float)

# 2-2  Morphologie du réseau vasculaire
def morph_feats(skel):
    lbl = label(skel>0)
    length = sum(p.perimeter for p in regionprops(lbl))
    neigh  = cv2.filter2D(skel, -1, np.ones((3,3), np.uint8))
    cnt    = ((neigh - skel)//255).astype(np.int32)
    ends   = np.sum((skel==255)&(cnt==1))
    joints = np.sum((skel==255)&(cnt>=3))
    return np.array([length, ends, joints], float)

# 2-3  Haralick / GLCM (contrast, corr, energy, homogeneity)
def haralick_feats(img, distances=(1,2,3), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    glcm  = greycomatrix(img, distances=distances, angles=angles,
                         symmetric=True, normed=True, levels=256)
    props = ['contrast','correlation','energy','homogeneity']
    return np.array([greycoprops(glcm, p).mean() for p in props], float)

# 2-4  HOG (gradients orientés)
def hog_feats(img, sz=(128,128)):
    img_r = cv2.resize(img, sz)
    h = hog(img_r, orientations=9, pixels_per_cell=(16,16),
            cells_per_block=(2,2), block_norm='L2-Hys',
            visualize=False, feature_vector=True)
    return h.astype(float)

# 2-5  Pipeline d'extraction complet
def extract_features(img,
                     use_c=True, use_d=True, use_v=True, use_m=True,
                     lbp_radii=(1,2,3)):
    proc = preprocess(img)
    if use_d: proc = denoise(proc)
    if use_c: proc = clahe(proc)
    if use_v: proc = vessels(proc)

    b = binarize(proc)
    if use_m: b = morph_close(b)
    s = skeleton(b)

    feats = np.concatenate([
        morph_feats(s),
        lbp_hist_multiscale(s, radii=lbp_radii),
        haralick_feats(proc),
        hog_feats(proc)
    ])

    return b, s, feats

# Pour compatibilité : ancien nom utilisé partout dans la GUI
# ------------------------------------------------------------------ #
# Pipeline complet : renvoie aussi une **image** LBP r=1             #
# ------------------------------------------------------------------ #
def pipeline_full(img, use_c, use_d, use_v, use_m):
    """
    Retourne :
        b  – binaire Otsu
        s  – squelette
        lbp_img – LBP uniforme r=1 (grayscale)
        feats   – vecteur de caractéristiques fusionné
    """
    # 1.  Pré-traitements
    proc = preprocess(img)
    if use_d: proc = denoise(proc)
    if use_c: proc = clahe(proc)
    if use_v: proc = vessels(proc)

    # 2.  Binarisation + morpho + squelette
    b = binarize(proc)
    if use_m: b = morph_close(b)
    s = skeleton(b)

    # 3.  Image LBP (uniform) r = 1  → même shape que s
    lbp_img = local_binary_pattern(s, 8, 1, method='uniform')
    lbp_img = ((lbp_img - lbp_img.min()) /
               (lbp_img.max() - lbp_img.min() + 1e-9) * 255).astype(np.uint8)

    # 4.  Features
    feats = np.concatenate([
        morph_feats(s),
        lbp_hist_multiscale(s, radii=(1,2,3)),
        haralick_feats(proc),
        hog_feats(proc)
    ])

    return b, s, lbp_img, feats

# -----------------------------------------------------------------------------#
# 3.  Modèles ML
# -----------------------------------------------------------------------------#
def build_model(name="LDA",
                lda_n_components=None,
                lda_solver="eigen",
                lda_shrinkage="auto",
                lda_tol=1e-4,
                lda_store_covariance=True):
    """
    Construit un pipeline de classification :
      - Pour 'LDA' : StandardScaler → PCA → CustomLDA
      
    Paramètres pour LDA :
      lda_n_components       : int ou None (nb composantes)
      lda_solver             : 'eigen' (votre implémentation)
      lda_shrinkage          : float, 'auto' ou None
      lda_tol                : float (tolérance pour la covariance)
      lda_store_covariance   : bool (stocker la cov)
    """
  
    if name == "LDA":
        return make_pipeline(
            StandardScaler(),
            # PCA conserve 98% de la variance ; changez si besoin
            PCA(n_components=0.98, whiten=True, random_state=42),
            CustomLDA(
                n_components=lda_n_components,
                solver=lda_solver,
                shrinkage=lda_shrinkage,
                tol=lda_tol,
                store_covariance=lda_store_covariance
            )
        )
    
    
    else:
        raise ValueError(f"Unknown model '{name}'")





def interpolate_curve(x, y, num_points=100):
    """
    Interpole une courbe sur un nombre fixe de points
    
    Args:
        x: Valeurs x de la courbe originale
        y: Valeurs y de la courbe originale
        num_points: Nombre de points pour l'interpolation
        
    Returns:
        x_new, y_new: Courbe interpolée
    """
    # Assurer que x est strictement croissant pour l'interpolation
    if len(x) < 2:
        # Si trop peu de points, retourner des valeurs par défaut
        return np.linspace(0, 1, num_points), np.zeros(num_points)
    
    # Gérer les valeurs dupliquées en x
    unique_indices = np.unique(x, return_index=True)[1]
    x_unique = x[unique_indices]
    y_unique = y[unique_indices]
    
    if len(x_unique) < 2:
        # Encore trop peu de points après déduplication
        return np.linspace(0, 1, num_points), np.zeros(num_points)
    
    # Assurer que x est strictement croissant
    sort_indices = np.argsort(x_unique)
    x_sorted = x_unique[sort_indices]
    y_sorted = y_unique[sort_indices]
    
    # Créer une fonction d'interpolation
    f = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value=(y_sorted[0], y_sorted[-1]))
    
    # Générer de nouveaux points x uniformément espacés
    x_new = np.linspace(0, 1, num_points)
    
    # Interpoler les valeurs y correspondantes
    y_new = f(x_new)
    
    return x_new, y_new

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate(model, X, y, labels, threshold=0.3):
    """
    Évalue le modèle sur les données fournies et calcule diverses métriques.
    
    Args:
        model: Modèle entraîné
        X: Caractéristiques
        y: Étiquettes réelles
        labels: Liste des étiquettes de classe
        threshold: Seuil pour la classification
        
    Returns:
        Dictionnaire contenant les résultats d'évaluation
    """
    try:
        yp = model.predict(X)
        rep_dict = classification_report(y, yp, output_dict=True, labels=labels, zero_division=0)
        rep_txt  = classification_report(y, yp, labels=labels, zero_division=0)
        cm       = confusion_matrix(y, yp, labels=labels)
        cv       = cross_val_score(model, X, y, cv=5).mean()

        eer_data = None
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)

                # Cas binaire
                if len(np.unique(y)) == 2:
                    target = np.unique(y)[1]
                    labels_bin = np.array([1 if label == target else 0 for label in y])
                    eer_data = compute_eer_data(scores, labels_bin)
            except Exception as e:
                print(f"[WARN] Erreur lors du calcul des données EER: {e}")

        return dict(rep_dict=rep_dict, rep_str=rep_txt, cm=cm, cv=cv,
                    labels=labels, yp=yp, eer_data=eer_data)
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'évaluation: {e}")
        return dict(error=str(e))




import numpy as np



def compute_eer_data(scores, labels_bin, steps=500):
    """
    Calcule les données TFA, TFR et TEE à partir des scores et des étiquettes binaires.
    
    Args:
        scores: Scores de confiance du modèle
        labels_bin: Étiquettes binaires (1 pour authentique, 0 pour imposteur)
        steps: Nombre de seuils à tester
        
    Returns:
        Dictionnaire contenant les courbes TFA, TFR et le point TEE
    """
    try:
        # Normalisation des scores avec protection contre division par zéro
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-10:  # Protection contre division par zéro
            scores_norm = np.zeros_like(scores)
        else:
            scores_norm = (scores - min_score) / (max_score - min_score) * 100

        # Calcul des seuils
        thresholds = np.linspace(0, 100, steps)
        tfa_list, tfr_list = [], []

        for thr in thresholds:
            preds = (scores_norm >= thr).astype(int)
            TP = np.sum((preds == 1) & (labels_bin == 1))
            FP = np.sum((preds == 1) & (labels_bin == 0))
            TN = np.sum((preds == 0) & (labels_bin == 0))
            FN = np.sum((preds == 0) & (labels_bin == 1))

            TFA = FP / (FP + TN) if (FP + TN) > 0 else 0  # Taux de Fausse Acceptation
            TFR = FN / (TP + FN) if (TP + FN) > 0 else 0  # Taux de Faux Rejet

            tfa_list.append(TFA)
            tfr_list.append(TFR)

        tfa_arr = np.array(tfa_list)
        tfr_arr = np.array(tfr_list)

        # Calcul du TEE (point où TFA = TFR)
        tee_idx = np.argmin(np.abs(tfa_arr - tfr_arr))
        tee = (tfa_arr[tee_idx] + tfr_arr[tee_idx]) / 2

        return {
            "fpr": tfa_arr,        # TFA (maintenu pour compatibilité)
            "frr": tfr_arr,        # TFR (maintenu pour compatibilité)
            "tfa": tfa_arr,        # TFA
            "tfr": tfr_arr,        # TFR
            "eer": tee,            # TEE (maintenu pour compatibilité)
            "tee": tee,            # TEE
            "eer_idx": tee_idx,    # Index du TEE (maintenu pour compatibilité)
            "tee_idx": tee_idx,    # Index du TEE
            "x_scale": thresholds, # Seuils normalisés (maintenu pour compatibilité)
            "seuils": thresholds   # Seuils normalisés
        }
    except Exception as e:
        print(f"[ERROR] Erreur dans compute_eer_data: {e}")
        return None



def compute_roc(model, X, y, labels):
    """
    Calcule la courbe ROC pour le modèle en utilisant model.decision_function()
    et en normalisant les scores comme dans compute_det_curve.
    
    Args:
        model: Modèle entraîné
        X: Caractéristiques
        y: Étiquettes réelles
        labels: Liste des étiquettes de classe
        
    Returns:
        Tuple (fpr_avg, tpr_avg, roc_auc) ou None si erreur
    """
    try:
        if not hasattr(model, "decision_function"):
            print("[WARN] Le modèle ne possède pas la méthode decision_function, impossible de calculer la courbe ROC")
            return None
        
        y_bin = label_binarize(y, classes=labels) #
        decision_scores_raw_all_classes = model.decision_function(X) # Scores bruts
        
        num_points = 100 #
        common_x = np.linspace(0, 1, num_points) #
        
        all_fpr_interp = np.zeros((len(labels), num_points)) #
        all_tpr_interp = np.zeros((len(labels), num_points)) #
        
        for i in range(len(labels)):
            class_scores_raw = decision_scores_raw_all_classes[:, i]
            
            # --- Début de la normalisation (identique à compute_det_curve) ---
            min_score = np.min(class_scores_raw)
            max_score = np.max(class_scores_raw)
            
            scores_norm_for_roc: np.ndarray # Type hint pour clarté
            if max_score - min_score < 1e-10: # Protection contre division par zéro
                scores_norm_for_roc = np.zeros_like(class_scores_raw)
            else:
                scores_norm_for_roc = (class_scores_raw - min_score) / (max_score - min_score) * 100 #
            # --- Fin de la normalisation ---

            # Utilisation des scores normalisés pour roc_curve
            fpr, tpr, _ = roc_curve(y_bin[:, i], scores_norm_for_roc) #
            
            fpr_interpolated_common, tpr_interpolated_common = interpolate_curve(fpr, tpr, num_points) #
            
            all_fpr_interp[i] = fpr_interpolated_common
            all_tpr_interp[i] = tpr_interpolated_common
        
        fpr_avg = np.mean(all_fpr_interp, axis=0) #
        tpr_avg = np.mean(all_tpr_interp, axis=0) #
        
        roc_auc = auc(fpr_avg, tpr_avg) #
        
        # Note : Même avec la normalisation des scores, la courbe ROC (fpr_avg, tpr_avg) et l'AUC (roc_auc)
        # devraient rester identiques à la version sans normalisation, car roc_curve est insensible
        # aux transformations linéaires des scores. La principale différence sera l'échelle des
        # seuils implicites considérés par roc_curve.

        return fpr_avg, tpr_avg, roc_auc
    except Exception as e:
        print(f"[ERROR] Erreur dans compute_roc: {e}") #
        import traceback
        traceback.print_exc()
        return None




from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize




import numpy as np

# ---------------------------------------------------------------------------- #
#  COMPUTE TFA / TFR + TEE
# ---------------------------------------------------------------------------- #
def compute_det_curve(model, X_test, y_test, labels, steps: int = 199,
                      norm_to_100: bool = True):
    """
    Calcule les courbes TFA (FAR), TFR (FRR) et le TEE global.

    Args
    ----
    model        : estimateur déjà entraîné (doit exposer decision_function
                   ou predict_proba pour produire un score de similarité)
    X_test       : matrice des caractéristiques de test   – shape (N, d)
    y_test       : étiquettes réelles                     – shape (N,)
    labels       : liste/array des classes présentes
    steps        : nombre de seuils (par défaut 199 → 0,5 % de résolution)
    norm_to_100  : True ⇒ les scores sont ramenés sur [0, 100] (utile si
                   vous mixez plusieurs modèles dont les plages diffèrent)

    Returns
    -------
    dict {
        'seuils'      : ndarray (steps,)   – échelle des seuils testés
        'tfa'         : ndarray (steps,)   – FAR   moyen
        'tfr'         : ndarray (steps,)   – FRR   moyen
        'tfa_per_cls' : ndarray (n_cls, steps)
        'tfr_per_cls' : ndarray (n_cls, steps)
        'tee'         : float              – taux d’égal-erreur
        'tee_idx'     : int                – index du TEE dans les courbes
    }
    """
    # ------------------------------------------------------------------ #
    # 1) Récupération des scores
    # ------------------------------------------------------------------ #
    if   hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        # On prend la log-proba de la classe vraisemblable pour conserver
        # une notion de marge (plus stable que la proba brute = [0,1]).
        raw_scores = np.log(model.predict_proba(X_test) + 1e-12)
    else:
        raise AttributeError("Le modèle ne fournit ni decision_function "
                             "ni predict_proba – impossible de tracer TFA/TFR.")

    raw_scores = np.atleast_2d(raw_scores)
    n_samples, n_cls = raw_scores.shape

    # Binaire ⇒ scikit renvoie shape (N,), on passe en (N,1)
    if n_cls == 1:
        raw_scores = np.column_stack([-raw_scores, raw_scores])
        n_cls = 2

    # ------------------------------------------------------------------ #
    # 2) Normalisation éventuelle à [0,100] (score ↑ ⇒ acceptation)
    # ------------------------------------------------------------------ #
    if norm_to_100:
        s_min = raw_scores.min(axis=0, keepdims=True)
        s_rng = raw_scores.max(axis=0, keepdims=True) - s_min
        s_rng[s_rng < 1e-12] = 1.0  # évite /0
        scores = (raw_scores - s_min) / s_rng * 100.0
        seuils = np.linspace(0, 100, steps)
    else:
        scores = raw_scores
        seuils = np.linspace(scores.min(), scores.max(), steps)

    # ------------------------------------------------------------------ #
    # 3) Construction des étiquettes binaires par classe
    # ------------------------------------------------------------------ #
    y_test = np.asarray(y_test)
    y_bin  = (y_test[:, None] == np.asarray(labels)[None, :]).astype(int)
    # y_bin shape = (N, n_cls)

    # ------------------------------------------------------------------ #
    # 4) Calcul vectorisé TFA / TFR pour chaque classe
    # ------------------------------------------------------------------ #
    # Pour chaque seuil, préd = (score ≥ seuil)  ⇒ shape (steps, N, n_cls)
    pred_mat = (scores[None, :, :] >= seuils[:, None, None])  # bool
    y_bin_t  = y_bin.T                                         # (n_cls, N)

    # TP = ∑ pred & y_bin,   FP = ∑ pred & ~y_bin,  etc.
    TP = np.einsum("sni,in->sn", pred_mat, y_bin_t)
    FP = np.einsum("sni,in->sn", pred_mat, 1 - y_bin_t)
    FN = np.einsum("sni,in->sn", 1 - pred_mat, y_bin_t)
    TN = np.einsum("sni,in->sn", 1 - pred_mat, 1 - y_bin_t)

    # TFA = FP / (FP + TN) ;  TFR = FN / (TP + FN)
    denom_tfa = FP + TN
    denom_tfr = TP + FN
    with np.errstate(divide="ignore", invalid="ignore"):
        tfa_cls = np.where(denom_tfa > 0, FP / denom_tfa, 0.0)
        tfr_cls = np.where(denom_tfr > 0, FN / denom_tfr, 0.0)

    # ------------------------------------------------------------------ #
    # 5) Agrégation sur toutes les classes (moyenne macro)
    # ------------------------------------------------------------------ #
    tfa_mean = tfa_cls.mean(axis=1)
    tfr_mean = tfr_cls.mean(axis=1)

    # ------------------------------------------------------------------ #
    # 6) TEE  (index où |TFA − TFR| est minimal)
    # ------------------------------------------------------------------ #
    tee_idx = int(np.argmin(np.abs(tfa_mean - tfr_mean)))
    tee     = (tfa_mean[tee_idx] + tfr_mean[tee_idx]) / 2.0

    # ------------------------------------------------------------------ #
    return {
        "seuils"      : seuils,
        "tfa"         : tfa_mean,
        "tfr"         : tfr_mean,
        "tfa_per_cls" : tfa_cls,          # shape (steps, n_cls)
        "tfr_per_cls" : tfr_cls,
        "tee"         : float(tee),
        "tee_idx"     : tee_idx,
    }





# ----------------------------------------------------------------------------- 
# Threads déportés
# -----------------------------------------------------------------------------
class LoaderThread(QThread):
    """
    Thread de chargement du dataset et d'entraînement des modèles.
    Émet :
      - progress(int) : progression de 0 à 100
      - finished(dict, dict) : g_rep et models
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object)

    def __init__(self, folder: str, opts: Tuple[bool, bool, bool, bool], parent=None):
        super().__init__(parent)
        self.folder = folder
        self.opts = opts  # (use_clahe, use_denoise, use_vessels, use_morph)

    def run(self):
        try:
            # 1) Récupération des fichiers images
            files = [
                f for f in sorted(os.listdir(self.folder))
                if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))
            ]
            X, y = [], []
            # 2) Extraction des features et mise à jour de la barre de progression
            for i, fname in enumerate(files):
                img = cv2.imread(os.path.join(self.folder, fname), cv2.IMREAD_GRAYSCALE)
                _, _, _, feats = pipeline_full(img, *self.opts)
                X.append(feats)
                # Extraction du label depuis le nom de fichier : ex. IMG_01_label.png
                label = fname.split('_')[1].split('.')[0].lower()
                y.append(label)
                self.progress.emit(int((i + 1) / len(files) * 100))

            X = np.array(X)
            y = np.array(y)
            labels = sorted(set(y))

            # 3) Entraînement et évaluation des modèles
            g_rep, models = {}, {}
            for name in ("LDA",):
                # Construction du pipeline LDA
                mdl = build_model(
                    name=name,
                    lda_n_components=None,
                    lda_solver="eigen",
                    lda_shrinkage="auto",
                    lda_tol=1e-4,
                    lda_store_covariance=True
                )
                mdl.fit(X, y)

                # Évaluation classique
                rep = evaluate(mdl, X, y, labels)

                # Courbe ROC
                roc = compute_roc(mdl, X, y, labels)
                if roc:
                    rep["roc"] = roc

                # Courbe DET (TFA/TFR/TEE)
                det = compute_det_curve(mdl, X, y, labels)
                if det:
                    rep["det"] = det

                g_rep[name] = rep
                models[name] = mdl

            # 4) Signal de fin avec résultats
            self.finished.emit(g_rep, models)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            # En cas d'erreur, on renvoie un dict d'erreur
            self.finished.emit({"error": str(e)}, {})

            

import time
import traceback # Pour un débogage plus détaillé
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# Supposons que pipeline_full est importée ou définie ailleurs dans pipeline.py
# from . import pipeline_full # Si c'est dans le même dossier et que __init__.py est configuré
# ou directement si définie dans le même fichier

class AnalyseThread(QThread):
    progress = pyqtSignal(int)
    done     = pyqtSignal(str)

    def __init__(self, path: str, model, opts: tuple, mode: str, claim: str, g_rep: dict, threshold: float = 0.5, parent=None):
        super().__init__(parent)
        self.path = path
        self.model = model
        self.opts = opts
        self.mode = mode
        self.claim = claim
        self.g_rep = g_rep
        self.threshold = threshold

    # --- Méthodes auxiliaires pour la construction des messages ---

    def _get_confidence_details(self, features: np.ndarray) -> tuple[float, str]:
        """
        Tente d'obtenir la probabilité maximale et la chaîne d'information sur la confiance.
        Retourne (max_probabilité, chaîne_info_confiance).
        max_probabilité est -1 si non disponible ou en cas d'erreur.
        """
        if hasattr(self.model, "predict_proba"):
            try:
                probas = self.model.predict_proba(features.reshape(1, -1))[0] #
                max_prob = np.max(probas) #
                return max_prob, f"\nScore de confiance: {max_prob:.3f}" #
            except Exception as e:
                print(f"Avertissement: Erreur lors du calcul de predict_proba: {e}")
                return -1, "\n(Erreur calcul score confiance)"
        return -1, "\n(Score de confiance non disponible)"

    def _build_identification_message(self, prediction: str, max_prob: float, confidence_info: str) -> str:
        """Construit le message pour le mode identification."""
        if max_prob != -1:  # Si les scores sont disponibles et valides
            if max_prob >= self.threshold:
                return f"Identification acceptée ✔ {confidence_info}\nIdentifié comme: {prediction}"
            else:
                return f"Échec d'identification (confiance faible) ❌ (Score: {max_prob:.3f}, < Seuil: {self.threshold:.3f})\nIdentifié comme: {prediction}"
        else: # Pas de scores valides disponibles
            return f"Identification → {prediction}{confidence_info}"

    def _build_authentication_message(self, prediction: str, max_prob: float, confidence_info: str) -> str:
        """Construit le message pour le mode authentification."""
        is_claim_correct = (prediction.lower() == self.claim.lower())

        if max_prob != -1:  # Si les scores sont disponibles et valides
            if max_prob >= self.threshold: # Score suffisant
                if is_claim_correct:
                    return f"Authentifié ✔{confidence_info}"
                else:
                    return f"Échec d'authentification ❌ (Identité incorrecte: {prediction} ≠ {self.claim.lower()}){confidence_info}"
            else: # Score insuffisant
                return f"Échec d'authentification (confiance faible) ❌ (Score: {max_prob:.3f}, < Seuil: {self.threshold:.3f})"
        else: # Pas de scores valides disponibles
            if is_claim_correct:
                return f"Authentifié ✔{confidence_info}"
            else:
                return f"Échec d'authentification ❌ ({prediction} ≠ {self.claim.lower()}){confidence_info}"

    # --- Méthode run() réorganisée ---

    def run(self):
        for i in range(101): 
            self.progress.emit(i)
            time.sleep(.01)
        
        final_message = "" # Initialiser pour éviter UnboundLocalError
        try:
            img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Il est préférable de lever une exception plus spécifique si possible
                raise FileNotFoundError(f"Impossible de charger l'image depuis : {self.path}") 
            
            # Assurez-vous que pipeline_full est accessible ici.
            # Si elle est dans le même fichier, l'appel direct est ok.
            # Sinon, vérifiez l'importation.
            _, _, _, feats = pipeline_full(img, *self.opts) 
            
            prediction = self.model.predict(feats.reshape(1, -1))[0] #
            
            max_prob, confidence_info_str = self._get_confidence_details(feats)

            if self.mode == "ident":
                final_message = self._build_identification_message(prediction, max_prob, confidence_info_str)
            else:  # self.mode == "auth"
                final_message = self._build_authentication_message(prediction, max_prob, confidence_info_str)
            
            # Ajouter les métriques biométriques globales
            # S'assurer que g_rep et ses clés existent avant d'y accéder
            if self.g_rep:
                if "identification_rate" in self.g_rep: #
                    final_message += f"\n\nTaux d'identification global (sur test): {self.g_rep['identification_rate']:.3f}"
                
                # Gestion plus sûre de l'accès à 'tee'
                det_data = self.g_rep.get("det", {})
                eer_data = self.g_rep.get("eer_data", {}) #

                if det_data and "tee" in det_data: #
                     final_message += f"\nTEE global (sur test): {det_data['tee']:.3f}"
                elif eer_data and "tee" in eer_data: #
                     final_message += f"\nTEE global (sur test): {eer_data['tee']:.3f}"
            
            self.done.emit(final_message)

        except FileNotFoundError as fnf_error:
            print(f"Erreur de fichier dans AnalyseThread: {fnf_error}")
            self.done.emit(f"Erreur d'analyse: {str(fnf_error)}")
        except ValueError as ve_error: # Pour d'autres erreurs de valeur potentielles
            print(f"Erreur de valeur dans AnalyseThread: {ve_error}")
            self.done.emit(f"Erreur d'analyse: {str(ve_error)}")
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Erreur inattendue dans AnalyseThread: {error_details}")
            # Éviter d'exposer les détails complets de traceback à l'utilisateur final via l'UI.
            self.done.emit(f"Erreur d'analyse inattendue. Vérifiez les logs console.") #


