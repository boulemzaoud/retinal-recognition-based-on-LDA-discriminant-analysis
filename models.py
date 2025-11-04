"""
Model registry – place ML pipelines here.
Implémentation améliorée avec les meilleures pratiques et sans fonctions prédéfinies.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg  # Utilisation de scipy.linalg pour la décomposition en valeurs propres généralisées
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class CustomLDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Implémentation personnalisée de l'Analyse Discriminante Linéaire (LDA).
    
    Cette classe implémente l'algorithme LDA sans utiliser la classe LinearDiscriminantAnalysis
    de scikit-learn. Elle calcule les projections discriminantes qui maximisent la séparation
    entre les classes tout en minimisant la variance au sein de chaque classe.
    """
    
    def __init__(self, n_components=None, solver="eigen", shrinkage=None, tol=1e-4, store_covariance=True):
        """
        Initialise l'analyse discriminante linéaire personnalisée.
        
        Args:
            n_components (int, optional): Nombre de composantes à conserver.
                Si None, utilise min(n_classes - 1, n_features).
            solver (str): Méthode de résolution ('eigen' uniquement supporté pour l'instant).
            shrinkage (float, str, optional): Paramètre de régularisation.
                Si 'auto', utilise la méthode de Ledoit-Wolf.
                Si float entre 0 et 1, utilise cette valeur directement.
                Si None, pas de régularisation.
            tol (float): Seuil de tolérance pour les valeurs propres.
            store_covariance (bool): Si True, stocke la matrice de covariance.
        
        Raises:
            ValueError: Si les paramètres ne sont pas valides.
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage
        self.tol = tol
        self.store_covariance = store_covariance
    
    def fit(self, X, y):
        """
        Ajuste le modèle LDA aux données d'entraînement.
        
        Args:
            X (array-like): Données d'entrée de forme (n_samples, n_features)
            y (array-like): Étiquettes de classe de forme (n_samples,)
            
        Returns:
            self: Objet ajusté
            
        Raises:
            ValueError: Si les données ne sont pas valides
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X doit être un tableau 2D")
        
        n_samples, n_features = X.shape
        
        # Identifier les classes uniques et leurs indices
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        
        if n_classes < 2:
            raise ValueError("Le nombre de classes doit être au moins 2")
        
        # Déterminer le nombre de composantes
        if self.n_components is None:
            self.n_components_ = min(n_classes - 1, n_features)
        else:
            self.n_components_ = min(self.n_components, n_classes - 1, n_features)
        
        # Calculer les probabilités a priori des classes
        self.priors_ = np.bincount(y_indices) / float(n_samples)
        
        # Calculer la moyenne globale
        self.xbar_ = np.mean(X, axis=0)
        
        # Calculer les moyennes par classe
        self.means_ = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            class_indices = np.where(y == self.classes_[i])[0]
            self.means_[i] = np.mean(X[class_indices], axis=0)
        
        # Calculer les matrices de dispersion
        Sw = np.zeros((n_features, n_features))  # Within-class scatter matrix
        Sb = np.zeros((n_features, n_features))  # Between-class scatter matrix
        
        # Calculer Sw (matrice de dispersion intra-classe)
        for i in range(n_classes):
            class_indices = np.where(y == self.classes_[i])[0]
            n_samples_class = len(class_indices)
            if n_samples_class <= 1:
                continue
                
            X_centered = X[class_indices] - self.means_[i]
            S_i = np.dot(X_centered.T, X_centered) / n_samples_class
            Sw += self.priors_[i] * S_i
        
        # Appliquer la régularisation si nécessaire
        if isinstance(self.shrinkage, str) and self.shrinkage.lower() == 'auto':
            # Méthode de Ledoit-Wolf pour estimer le paramètre de régularisation
            shrinkage = self._ledoit_wolf_shrinkage(X, y, Sw)
        elif self.shrinkage is not None:
            shrinkage = float(self.shrinkage)
        else:
            shrinkage = 0
            
        if shrinkage > 0:
            # Régularisation: (1-shrinkage)*Sw + shrinkage*np.eye(n_features)*np.trace(Sw)/n_features
            shrinkage_target = np.eye(n_features) * np.trace(Sw) / n_features
            Sw = (1 - shrinkage) * Sw + shrinkage * shrinkage_target
        
        # Ajouter une petite régularisation pour éviter les problèmes numériques
        Sw += np.eye(n_features) * self.tol
        
        # Stocker la matrice de covariance si demandé
        if self.store_covariance:
            self.covariance_ = Sw.copy()
        
        # Calculer Sb (matrice de dispersion inter-classe)
        for i in range(n_classes):
            mean_diff = self.means_[i] - self.xbar_
            Sb += self.priors_[i] * np.outer(mean_diff, mean_diff)
        
        # Résoudre le problème des valeurs propres généralisées
        try:
            # Utiliser scipy.linalg.eigh pour le problème généralisé
            evals, evecs = linalg.eigh(Sb, Sw)
        except np.linalg.LinAlgError as e:
            # En cas d'erreur, ajouter une régularisation plus forte et réessayer
            Sw += np.eye(n_features) * 0.01
            evals, evecs = linalg.eigh(Sb, Sw)
        
        # Trier les valeurs propres et vecteurs propres par ordre décroissant
        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]
        
        # Conserver uniquement les n_components premières composantes
        self.scalings_ = evecs[:, :self.n_components_]
        
        # Normaliser les vecteurs propres
        for i in range(self.scalings_.shape[1]):
            self.scalings_[:, i] = self.scalings_[:, i] / np.linalg.norm(self.scalings_[:, i])
        
        # Calculer les ratios de variance expliquée
        total_var = np.sum(evals)
        if total_var > 0:
            self.explained_variance_ratio_ = evals[:self.n_components_] / total_var
        else:
            self.explained_variance_ratio_ = np.ones(self.n_components_) / self.n_components_
        
        # Calculer les coefficients pour la prédiction
        self._compute_coef()
        
        return self
    
    def _ledoit_wolf_shrinkage(self, X, y, Sw):
        """
        Calcule le paramètre de régularisation optimal selon la méthode de Ledoit-Wolf.
        
        Args:
            X (array-like): Données d'entrée
            y (array-like): Étiquettes de classe
            Sw (array-like): Matrice de dispersion intra-classe
            
        Returns:
            float: Paramètre de régularisation optimal
        """
        n_samples, n_features = X.shape
        
        # Estimation simplifiée du paramètre de régularisation
        # Cette implémentation est une approximation de la méthode de Ledoit-Wolf
        mu = np.trace(Sw) / n_features
        
        # Éviter la division par zéro
        if mu == 0:
            return 0.5  # Valeur par défaut raisonnable
            
        delta = np.mean((Sw - mu * np.eye(n_features))**2)
        
        # Calculer la variance des données centrées par classe
        X_centered = np.zeros_like(X)
        for i, cls in enumerate(self.classes_):
            mask = (y == cls)
            X_centered[mask] = X[mask] - self.means_[i]
        
        # Estimer la variance d'échantillonnage
        var_sample = np.mean(X_centered**2)
        
        # Éviter la division par zéro
        if var_sample == 0:
            return 0.5  # Valeur par défaut raisonnable
            
        # Calculer le paramètre de régularisation
        alpha = min(delta / var_sample, 1.0)
        return alpha
    
    def _compute_coef(self):
        """
        Calcule les coefficients pour la prédiction.
        """
        n_classes = len(self.classes_)
        n_features = self.means_.shape[1]
        
        # Calculer les coefficients et les intercepts pour chaque classe
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        
        # Transformation des moyennes de classe
        transformed_means = np.dot(self.means_, self.scalings_)
        
        for i in range(n_classes):
            # Calculer les coefficients pour la classification
            self.coef_[i] = 2 * np.dot(self.scalings_, transformed_means[i])
            # Calculer les intercepts
            self.intercept_[i] = -np.sum(transformed_means[i] ** 2) + 2 * np.log(self.priors_[i])
    
    def transform(self, X):
        """
        Projette les données sur les composantes discriminantes.
        
        Args:
            X (array-like): Données à projeter de forme (n_samples, n_features)
            
        Returns:
            array-like: Données projetées de forme (n_samples, n_components)
        """
        X = np.asarray(X)
        return np.dot(X, self.scalings_)
    
    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée de forme (n_samples, n_features)
            
        Returns:
            array-like: Classes prédites de forme (n_samples,)
        """
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
    
    def predict_proba(self, X):
        """
        Prédit les probabilités de classe pour les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée de forme (n_samples, n_features)
            
        Returns:
            array-like: Probabilités de classe de forme (n_samples, n_classes)
        """
        scores = self.decision_function(X)
        
        # Convertir les scores en probabilités via softmax
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        
        return proba
    
    def decision_function(self, X):
        """
        Calcule la fonction de décision pour les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée de forme (n_samples, n_features)
            
        Returns:
            array-like: Valeurs de la fonction de décision de forme (n_samples, n_classes)
        """
        X = np.asarray(X)
        
        # Utiliser directement les coefficients calculés pour la prédiction
        return np.dot(X, self.coef_.T) + self.intercept_
    
    def score(self, X, y):
        """
        Calcule le score de précision sur les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée de forme (n_samples, n_features)
            y (array-like): Étiquettes réelles de forme (n_samples,)
            
        Returns:
            float: Score de précision
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def fit_transform(self, X, y):
        """
        Ajuste le modèle aux données puis transforme les données.
        
        Args:
            X (array-like): Données d'entrée de forme (n_samples, n_features)
            y (array-like): Étiquettes de classe de forme (n_samples,)
            
        Returns:
            array-like: Données transformées de forme (n_samples, n_components)
        """
        self.fit(X, y)
        return self.transform(X)


class Pipeline(BaseEstimator):
    """
    Classe personnalisée pour créer un pipeline de transformations et d'estimateurs.
    
    Cette classe permet d'enchaîner plusieurs étapes de transformation et d'estimation
    sans utiliser la fonction prédéfinie make_pipeline de scikit-learn.
    """
    
    def __init__(self, steps):
        """
        Initialise un pipeline avec une séquence d'étapes.
        
        Args:
            steps (list): Liste de tuples (nom, transformateur/estimateur)
        
        Raises:
            ValueError: Si la liste des étapes est vide
        """
        if not steps:
            raise ValueError("Le pipeline doit contenir au moins une étape")
        
        self.steps = steps
        self._validate_steps()
    
    def _validate_steps(self):
        """
        Valide les étapes du pipeline.
        
        Raises:
            TypeError: Si les noms d'étapes ne sont pas des chaînes
            ValueError: Si les noms d'étapes ne sont pas uniques
        """
        names, estimators = zip(*self.steps)
        
        # Vérifier que les noms sont des chaînes
        if not all(isinstance(name, str) for name in names):
            raise TypeError("Les noms d'étapes doivent être des chaînes")
        
        # Vérifier que les noms sont uniques
        if len(set(names)) != len(names):
            raise ValueError("Les noms d'étapes doivent être uniques")
        
        # Créer un dictionnaire des étapes nommées
        self.named_steps = dict(self.steps)
    
    def get_params(self, deep=True):
        """
        Obtient les paramètres du pipeline pour la validation croisée.
        
        Args:
            deep (bool): Si True, retourne également les paramètres des estimateurs contenus.
            
        Returns:
            dict: Dictionnaire des paramètres
        """
        params = {'steps': self.steps}
        
        if deep:
            for name, estimator in self.steps:
                if hasattr(estimator, 'get_params'):
                    for key, value in estimator.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
        
        return params
    
    def set_params(self, **params):
        """
        Définit les paramètres du pipeline.
        
        Args:
            **params: Paramètres à définir
            
        Returns:
            self: Pipeline avec les paramètres mis à jour
        """
        # Paramètres pour les étapes
        step_params = {}
        for key, value in params.items():
            if key == 'steps':
                self.steps = value
                self._validate_steps()
            elif '__' in key:
                step, param = key.split('__', 1)
                if step not in step_params:
                    step_params[step] = {}
                step_params[step][param] = value
        
        # Mettre à jour les paramètres des estimateurs
        for step, param_dict in step_params.items():
            if step in self.named_steps:
                estimator = self.named_steps[step]
                if hasattr(estimator, 'set_params'):
                    estimator.set_params(**param_dict)
        
        return self
    
    def fit(self, X, y=None):
        """
        Ajuste le pipeline aux données d'entraînement.
        
        Args:
            X (array-like): Données d'entrée pour l'entraînement
            y (array-like, optional): Cibles pour l'entraînement
            
        Returns:
            self: Pipeline ajusté
        """
        X_transformed = X
        
        for name, transformer in self.steps[:-1]:
            X_transformed = self._fit_transform_one(transformer, X_transformed, y)
        
        # Ajuster le dernier estimateur
        if y is not None:
            self.steps[-1][1].fit(X_transformed, y)
        else:
            self.steps[-1][1].fit(X_transformed)
            
        return self
    
    def _fit_transform_one(self, transformer, X, y=None):
        """
        Ajuste un transformateur et applique la transformation.
        
        Args:
            transformer: Transformateur à ajuster
            X (array-like): Données d'entrée
            y (array-like, optional): Cibles
            
        Returns:
            array-like: Données transformées
        """
        if y is not None and hasattr(transformer, 'fit_transform'):
            return transformer.fit_transform(X, y)
        else:
            return transformer.fit(X).transform(X)
    
    def transform(self, X):
        """
        Applique les transformations aux données.
        
        Args:
            X (array-like): Données à transformer
            
        Returns:
            array-like: Données transformées
        """
        X_transformed = X
        
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)
            
        return X_transformed
    
    def predict(self, X):
        """
        Prédit les cibles pour les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée
            
        Returns:
            array-like: Prédictions
        """
        X_transformed = self.transform(X)
        return self.steps[-1][1].predict(X_transformed)
    
    def predict_proba(self, X):
        """
        Prédit les probabilités pour les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée
            
        Returns:
            array-like: Probabilités prédites
        """
        X_transformed = self.transform(X)
        return self.steps[-1][1].predict_proba(X_transformed)
    
    def decision_function(self, X):
        """
        Calcule la fonction de décision pour les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée
            
        Returns:
            array-like: Valeurs de la fonction de décision
        """
        X_transformed = self.transform(X)
        return self.steps[-1][1].decision_function(X_transformed)
    
    def score(self, X, y):
        """
        Calcule le score de précision sur les données d'entrée.
        
        Args:
            X (array-like): Données d'entrée
            y (array-like): Cibles réelles
            
        Returns:
            float: Score de précision
        """
        X_transformed = self.transform(X)
        return self.steps[-1][1].score(X_transformed, y)


def lda_pipeline(n_components=None):
    """
    Crée et retourne un pipeline personnalisé:
    StandardScaler ➜ PCA(95% var) ➜ CustomLDA(shrinkage='auto')
    
    Ce pipeline standardise d'abord les données, puis applique une réduction
    de dimensionnalité par PCA conservant 95% de la variance, et enfin
    effectue une analyse discriminante linéaire avec régularisation automatique.
    
    Args:
        n_components (int, optional): Nombre de composantes à conserver pour LDA.
            Si None, utilise min(n_classes - 1, n_features).
    
    Returns:
        Pipeline: Pipeline personnalisé pour la classification
        
    Example:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> pipeline = lda_pipeline()
        >>> pipeline.fit(X_train, y_train)
        >>> accuracy = pipeline.score(X_test, y_test)
    """
    # Création des étapes du pipeline
    scaler = StandardScaler()
    pca = PCA(n_components=0.95, whiten=True)
    lda = CustomLDA(n_components=n_components, shrinkage="auto")  # Utilisation de notre LDA personnalisée
    
    # Construction du pipeline personnalisé
    steps = [
        ('scaler', scaler),
        ('pca', pca),
        ('lda', lda)
    ]
    
    return Pipeline(steps)


