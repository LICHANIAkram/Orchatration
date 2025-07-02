<<<<<<< HEAD
# ==========================================================
# 0. Imports
# ==========================================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 1. Chargement et analyse du jeu de données
# ----------------------------------------------------------
df = pd.read_csv("Data_Set.csv")
print("Shape des données:", df.shape)
print("\nDistribution des classes:")
print(df["algorithm"].value_counts())
print("\nInformations sur les données:")
print(df.info())

# ----------------------------------------------------------
# 2. Feature Engineering avancé
# ----------------------------------------------------------
def create_advanced_features(df):
    """Créer des features plus sophistiquées"""
    df_new = df.copy()
    
    # Features existantes améliorées
    df_new["Size_sensitivity_Ratio"] = df_new["size"] / (df_new["sensitivity"] + 1e-8)
    df_new["Log_Size"] = np.log1p(df_new["size"])
    df_new["Size_sensitivity_Product"] = df_new["size"] * df_new["sensitivity"]
    
    # Nouvelles features statistiques
    df_new["Sqrt_Size"] = np.sqrt(df_new["size"])
    df_new["Size_Squared"] = df_new["size"] ** 2
    df_new["Size_Cubed"] = df_new["size"] ** 3
    df_new["Size_sensitivity_Diff"] = abs(df_new["size"] - df_new["sensitivity"] * 100)
    
    # Features d'interaction complexes
    df_new["Size_x_sensitivity_x_Log"] = df_new["size"] * df_new["sensitivity"] * df_new["Log_Size"]
    df_new["Size_Log_Ratio"] = df_new["size"] / (df_new["Log_Size"] + 1e-8)
    df_new["Sensitivity_Squared"] = df_new["sensitivity"] ** 2
    
    # Éviter les valeurs trop grandes pour Size_Sensitivity_Power
    df_new["Size_Sensitivity_Power"] = np.where(
        df_new["sensitivity"] > 2, 
        df_new["size"] ** 2,  # Limiter à puissance 2 si sensitivity > 2
        df_new["size"] ** df_new["sensitivity"]
    )
    
    # Features binaires et catégorielles
    df_new["Is_Large_File"] = (df_new["size"] > df_new["size"].quantile(0.75)).astype(int)
    df_new["Is_Small_File"] = (df_new["size"] < df_new["size"].quantile(0.25)).astype(int)
    df_new["Is_High_Sensitivity"] = (df_new["sensitivity"] > df_new["sensitivity"].median()).astype(int)
    
    # Features de rang
    df_new["Size_Rank"] = df_new["size"].rank(pct=True)
    df_new["Sensitivity_Rank"] = df_new["sensitivity"].rank(pct=True)
    
    return df_new

df = create_advanced_features(df)

# ----------------------------------------------------------
# 3. Preprocessing amélioré
# ----------------------------------------------------------
# Suppression des colonnes inutiles
df.drop(columns=["name"], inplace=True)

# Encodage avec gestion des nouvelles catégories
label_encoders = {}

le_type = LabelEncoder()
df["type"] = le_type.fit_transform(df["type"])
label_encoders["type"] = le_type

le_extension = LabelEncoder()
df["extension"] = le_extension.fit_transform(df["extension"])
label_encoders["extension"] = le_extension

target_encoder = LabelEncoder()
df["algorithm"] = target_encoder.fit_transform(df["algorithm"])

# ----------------------------------------------------------
# 4. Sélection des features optimisée
# ----------------------------------------------------------
feature_cols = [
    "type", "extension", "size", "sensitivity",
    "Size_sensitivity_Ratio", "Log_Size", "Size_sensitivity_Product",
    "Sqrt_Size", "Size_Squared", "Size_Cubed", "Size_sensitivity_Diff",
    "Size_x_sensitivity_x_Log", "Size_Log_Ratio", "Sensitivity_Squared",
    "Size_Sensitivity_Power", "Is_Large_File", "Is_Small_File", 
    "Is_High_Sensitivity", "Size_Rank", "Sensitivity_Rank"
]

X = df[feature_cols]
y = df["algorithm"]

# Gestion des valeurs infinies et NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# ----------------------------------------------------------
# 5. Scaling robuste
# ----------------------------------------------------------
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# 6. Stratégie d'oversampling améliorée
# ----------------------------------------------------------
categorical_features = [0, 1]  # type et extension

# SMOTEENN combine over et under-sampling
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

print(f"Taille après resampling: {X_resampled.shape}")
print("Distribution après resampling:")
unique, counts = np.unique(y_resampled, return_counts=True)
for i, count in zip(unique, counts):
    print(f"  Classe {target_encoder.classes_[i]}: {count}")

# ----------------------------------------------------------
# 7. Split stratifié
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.20, random_state=42, stratify=y_resampled
)

X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTailles des splits:")
print(f"Train fit: {X_train_fit.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# ----------------------------------------------------------
# 7b. Validation croisée et recherche d'hyperparamètres
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    'depth': [4, 6],
    'learning_rate': [0.03, 0.05],
    'l2_leaf_reg': [3, 5],
    'iterations': [1000]
}

base_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=0,
    random_seed=42
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print("\nRecherche d'hyperparamètres (peut être long)...")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Meilleurs paramètres: {best_params}")

# ----------------------------------------------------------
# 8. Configuration CatBoost stabilisée
# ----------------------------------------------------------
def create_stable_catboost(params=None):
    """Configuration CatBoost optimisée pour la stabilité"""

    if params is None:
        params = {}
    
    model = CatBoostClassifier(
        # Paramètres pour la stabilité
        iterations=params.get('iterations', 1200),
        learning_rate=params.get('learning_rate', 0.03),
        depth=params.get('depth', 5),
        l2_leaf_reg=params.get('l2_leaf_reg', 5),
        bagging_temperature=0.8,            # Randomisation contrôlée
        random_strength=0.8,                # Force de randomisation réduite
        
        # Paramètres de robustesse
        bootstrap_type='Bayesian',          # Bootstrap bayésien plus stable
        sampling_frequency='PerTreeLevel',  # Échantillonnage par niveau
        leaf_estimation_method='Newton',    # Méthode d'estimation plus stable
        grow_policy='SymmetricTree',        # Croissance symétrique
        
        # Paramètres spécifiques
        border_count=32,                    # Moins de frontières pour plus de stabilité
        feature_border_type='Median',       # Frontières basées sur la médiane
        
        # Configuration générale
        loss_function="MultiClass",
        eval_metric="Accuracy",
        use_best_model=True,
        od_type='Iter',                     # Early stopping par itération
        od_wait=100,                        # Patience augmentée
        random_seed=42,
        verbose=50
    )
    
    return model

# ----------------------------------------------------------
# 9. Fonctions utilitaires pour analyse
# ----------------------------------------------------------
def plot_stable_learning_curves(model, X_train, X_val, y_train, y_val):
    """Visualisation des courbes d'apprentissage lissées"""
    
    # Entraîner avec logging détaillé
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=100,
        verbose=0
    )
    
    # Récupérer les métriques
    evals = model.get_evals_result()

    # Certaines versions de CatBoost nomment l'ensemble de validation
    # "validation_0" au lieu de "validation". On gère donc les deux cas.
    val_key = 'validation'
    if val_key not in evals:
        # Prendre la première clé autre que "learn" comme validation
        val_key = [k for k in evals if k != 'learn'][0]

    plt.figure(figsize=(15, 5))

    # Original curves
    plt.subplot(1, 3, 1)
    train_scores = evals['learn']['Accuracy']
    val_scores = evals[val_key]['Accuracy']
    
    plt.plot(train_scores, label='Train', alpha=0.7)
    plt.plot(val_scores, label='Validation', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves - Original')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Smoothed curves (moving average)
    plt.subplot(1, 3, 2)
    window = 10
    train_smooth = pd.Series(train_scores).rolling(window=window).mean()
    val_smooth = pd.Series(val_scores).rolling(window=window).mean()
    
    plt.plot(train_smooth, label='Train (lissé)', linewidth=2)
    plt.plot(val_smooth, label='Validation (lissé)', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(f'Smoothed Curves (window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Stability analysis
    plt.subplot(1, 3, 3)
    val_diff = np.diff(val_scores)
    plt.plot(val_diff, alpha=0.7, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Δ Accuracy')
    plt.title('Validation Accuracy Changes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("stable_learning_curves.png")
    plt.close()
    
    # Statistiques de stabilité
    val_stability = np.std(val_scores[-100:])  # Stabilité sur les 100 dernières itérations
    print(f"Stabilité de validation (100 dernières itérations): {val_stability:.6f}")
    
    return val_stability

def analyze_errors(y_true, y_pred, target_encoder):
    """Analyser les erreurs de classification"""
    
    # Matrice de confusion détaillée
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nAnalyse des erreurs par classe:")
    for i, class_name in enumerate(target_encoder.classes_):
        total_actual = np.sum(cm[i, :])
        correct_pred = cm[i, i]
        error_rate = 1 - (correct_pred / total_actual) if total_actual > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  Taux d'erreur: {error_rate:.3f}")
        print(f"  Prédictions correctes: {correct_pred}/{total_actual}")
        
        # Classes les plus confondues
        if total_actual > 0:
            errors = cm[i, :].copy()
            errors[i] = 0  # Enlever les bonnes prédictions
            max_error_idx = np.argmax(errors)
            if errors[max_error_idx] > 0:
                print(f"  Souvent confondu avec: {target_encoder.classes_[max_error_idx]} ({errors[max_error_idx]} fois)")


def plot_learning_curve_ensemble(model, X, y):
    """Tracer la courbe d'apprentissage pour l'ensemble"""

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, "o-", label="Train", color="blue")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="blue",
    )
    plt.plot(train_sizes, val_mean, "o-", label="Validation", color="green")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color="green",
    )
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve_ensemble.eps", format="eps")
    plt.savefig("learning_curve_ensemble.png")
    plt.close()

# ----------------------------------------------------------
# 10. Entraînement du modèle stabilisé
# ----------------------------------------------------------
print("=== ENTRAÎNEMENT CATBOOST STABILISÉ ===")

# Créer le modèle stabilisé avec les meilleurs paramètres de la grille
model = create_stable_catboost(best_params)

print("Entraînement du modèle avec early stopping...")
model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    plot=False
)

# Vérification de stabilité
eval_metrics = model.get_evals_result()
val_key = 'validation'
if val_key not in eval_metrics:
    val_key = [k for k in eval_metrics if k != 'learn'][0]
val_scores = eval_metrics[val_key]['Accuracy']
stability = np.std(val_scores[-100:])
print(f"\nStabilité (écart-type sur 100 dernières itérations): {stability:.6f}")

if stability < 0.005:
    print("✅ Modèle stable")
else:
    print("⚠️ Modèle instable - considérer réduire learning_rate")

# ----------------------------------------------------------
# 11. Ensemble de modèles (optionnel)
# ----------------------------------------------------------
def create_ensemble():
    """Créer un ensemble de modèles"""
    
    # CatBoost pour l'ensemble
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=15,
        l2_leaf_reg=3,
        bootstrap_type='Bayesian',
        loss_function="MultiClass",
        eval_metric="Accuracy",
        use_best_model=False,  # Important: False pour l'ensemble
        random_seed=42,
        verbose=0
    )
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    
    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000, random_state=42, multi_class='ovr'
    )
    
    # Ensemble voting
    ensemble = VotingClassifier([
        ('catboost', cb_model),
        ('random_forest', rf_model),
        ('logistic', lr_model)
    ], voting='soft', n_jobs=1)
    
    print("Entraînement de l'ensemble de modèles...")
    ensemble.fit(X_train, y_train)
    return ensemble

print("\nCréation de l'ensemble de modèles...")
ensemble_model = create_ensemble()

# ----------------------------------------------------------
# 12. Évaluation comparative
# ----------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """Évaluer un modèle"""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
    
    print(f"\n=== {model_name} ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1‑score  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    
    return y_pred

# Évaluation des modèles
y_pred_single = evaluate_model(model, X_test, y_test, "CatBoost Stabilisé")
y_pred_ensemble = evaluate_model(ensemble_model, X_test, y_test, "Ensemble de Modèles")

# Rapport détaillé pour le meilleur modèle
print("\nRapport de classification détaillé (CatBoost Stabilisé):")
print(classification_report(
    y_test, y_pred_single, target_names=target_encoder.classes_, zero_division=0
))

print("\nRapport de classification détaillé (Ensemble de Modèles):")
print(classification_report(
    y_test, y_pred_ensemble, target_names=target_encoder.classes_, zero_division=0
))

# ----------------------------------------------------------
# 13. Visualisations améliorées
# ----------------------------------------------------------
# Importance des features CatBoost
plt.figure(figsize=(6, 4))

feature_importance = model.get_feature_importance()
sorted_idx = np.argsort(feature_importance)[-15:]
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
plt.xlabel('Importance')
plt.title('Top 15 Features - Stable CatBoost')
plt.tight_layout()
plt.savefig('feature_importance_catboost.eps', format='eps')
plt.savefig('feature_importance_catboost.png')
plt.close()

# Matrice de confusion - CatBoost
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_single)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(cmap="Blues", xticks_rotation='horizontal')
plt.title(
    f"Confusion Matrix - Stable CatBoost (Acc: {accuracy_score(y_test, y_pred_single):.3f})"
)
plt.tight_layout()
plt.savefig('confusion_matrix_catboost.eps', format='eps')
plt.savefig('confusion_matrix_catboost.png')
plt.close()

# Matrice de confusion - Ensemble
plt.figure(figsize=(6, 4))
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
disp_ensemble = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=target_encoder.classes_)
disp_ensemble.plot(cmap="Greens", xticks_rotation='horizontal')
plt.title(
    f"Confusion Matrix - Ensemble (Acc: {accuracy_score(y_test, y_pred_ensemble):.3f})"
)
plt.tight_layout()
plt.savefig('confusion_matrix_ensemble.eps', format='eps')
plt.savefig('confusion_matrix_ensemble.png')
plt.close()

# Courbes d'apprentissage - CatBoost
plt.figure(figsize=(6, 4))

eval_metrics = model.get_evals_result()
train_acc = eval_metrics['learn']['Accuracy']
val_key = 'validation'
if val_key not in eval_metrics:
    val_key = [k for k in eval_metrics if k != 'learn'][0]
val_acc = eval_metrics[val_key]['Accuracy']

window = 20
train_smooth = pd.Series(train_acc).rolling(window=window, min_periods=1).mean()
val_smooth = pd.Series(val_acc).rolling(window=window, min_periods=1).mean()
plt.plot(train_acc, alpha=0.3, color='blue', label='Train (original)')
plt.plot(val_acc, alpha=0.3, color='red', label='Validation (original)')
plt.plot(train_smooth, color='blue', linewidth=2, label='Train (lissé)')
plt.plot(val_smooth, color='red', linewidth=2, label='Validation (lissé)')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title("CatBoost Learning Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve_catboost.eps', format='eps')
plt.savefig('learning_curve_catboost.png')
plt.close()

# Courbes d'apprentissage - Ensemble
plot_learning_curve_ensemble(ensemble_model, X_train, y_train)

# Performance comparison
plt.figure(figsize=(6, 4))

models = ['Stable CatBoost', 'Ensemble']
accuracies = [accuracy_score(y_test, y_pred_single), accuracy_score(y_test, y_pred_ensemble)]
colors = ['blue', 'green']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('model_comparison.eps', format='eps')
plt.savefig('model_comparison.png')
plt.close()

# ----------------------------------------------------------
# 14. Analyse détaillée des erreurs
# ----------------------------------------------------------
#analyze_errors(y_test, y_pred_single, target_encoder)

# ----------------------------------------------------------
# 15. Analyse de stabilité approfondie
# ----------------------------------------------------------
print("\n=== ANALYSE DE STABILITÉ ===")
stability_model = create_stable_catboost()
plot_stable_learning_curves(stability_model, X_train_fit, X_val, y_train_fit, y_val)

# ----------------------------------------------------------
# 16. Sauvegarde des modèles
# ----------------------------------------------------------
model.save_model("catboost_stable_model.cbm")
joblib.dump(ensemble_model, "ensemble_stable_model.pkl")
joblib.dump(scaler, "robust_scaler.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n=== MODÈLES SAUVEGARDÉS ===")
print("Fichiers créés:")
print("- catboost_stable_model.cbm")
print("- ensemble_stable_model.pkl")
print("- robust_scaler.pkl")
print("- target_encoder.pkl")
print("- label_encoders.pkl")
print("- feature_importance_catboost.eps")
print("- feature_importance_catboost.png")
print("- confusion_matrix_catboost.eps")
print("- confusion_matrix_catboost.png")
print("- confusion_matrix_ensemble.eps")
print("- confusion_matrix_ensemble.png")
print("- learning_curve_catboost.eps")
print("- learning_curve_catboost.png")
print("- learning_curve_ensemble.eps")
print("- learning_curve_ensemble.png")
print("- model_comparison.eps")
print("- model_comparison.png")

print(f"\n=== RÉSUMÉ FINAL ===")
print(f"Stabilité du modèle: {stability:.6f} {'✅' if stability < 0.005 else '⚠️'}")
print(f"Accuracy CatBoost Stabilisé: {accuracy_score(y_test, y_pred_single):.4f}")
print(f"Accuracy Ensemble: {accuracy_score(y_test, y_pred_ensemble):.4f}")
=======
# ==========================================================
# 0. Imports
# ==========================================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTENC, ADASYN
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 1. Chargement et analyse du jeu de données
# ----------------------------------------------------------
df = pd.read_csv("Data_Set.csv")
print("Shape des données:", df.shape)
print("\nDistribution des classes:")
print(df["algorithm"].value_counts())
print("\nInformations sur les données:")
print(df.info())

# ----------------------------------------------------------
# 2. Feature Engineering avancé
# ----------------------------------------------------------
def create_advanced_features(df):
    """Créer des features plus sophistiquées"""
    df_new = df.copy()
    
    # Features existantes améliorées
    df_new["Size_sensitivity_Ratio"] = df_new["size"] / (df_new["sensitivity"] + 1e-8)
    df_new["Log_Size"] = np.log1p(df_new["size"])
    df_new["Size_sensitivity_Product"] = df_new["size"] * df_new["sensitivity"]
    
    # Nouvelles features statistiques
    df_new["Sqrt_Size"] = np.sqrt(df_new["size"])
    df_new["Size_Squared"] = df_new["size"] ** 2
    df_new["Size_Cubed"] = df_new["size"] ** 3
    df_new["Size_sensitivity_Diff"] = abs(df_new["size"] - df_new["sensitivity"] * 100)
    
    # Features d'interaction complexes
    df_new["Size_x_sensitivity_x_Log"] = df_new["size"] * df_new["sensitivity"] * df_new["Log_Size"]
    df_new["Size_Log_Ratio"] = df_new["size"] / (df_new["Log_Size"] + 1e-8)
    df_new["Sensitivity_Squared"] = df_new["sensitivity"] ** 2
    
    # Éviter les valeurs trop grandes pour Size_Sensitivity_Power
    df_new["Size_Sensitivity_Power"] = np.where(
        df_new["sensitivity"] > 2, 
        df_new["size"] ** 2,  # Limiter à puissance 2 si sensitivity > 2
        df_new["size"] ** df_new["sensitivity"]
    )
    
    # Features binaires et catégorielles
    df_new["Is_Large_File"] = (df_new["size"] > df_new["size"].quantile(0.75)).astype(int)
    df_new["Is_Small_File"] = (df_new["size"] < df_new["size"].quantile(0.25)).astype(int)
    df_new["Is_High_Sensitivity"] = (df_new["sensitivity"] > df_new["sensitivity"].median()).astype(int)
    
    # Features de rang
    df_new["Size_Rank"] = df_new["size"].rank(pct=True)
    df_new["Sensitivity_Rank"] = df_new["sensitivity"].rank(pct=True)
    
    return df_new

df = create_advanced_features(df)

# ----------------------------------------------------------
# 3. Preprocessing amélioré
# ----------------------------------------------------------
# Suppression des colonnes inutiles
df.drop(columns=["name"], inplace=True)

# Encodage avec gestion des nouvelles catégories
label_encoders = {}

le_type = LabelEncoder()
df["type"] = le_type.fit_transform(df["type"])
label_encoders["type"] = le_type

le_extension = LabelEncoder()
df["extension"] = le_extension.fit_transform(df["extension"])
label_encoders["extension"] = le_extension

target_encoder = LabelEncoder()
df["algorithm"] = target_encoder.fit_transform(df["algorithm"])

# ----------------------------------------------------------
# 4. Sélection des features optimisée
# ----------------------------------------------------------
feature_cols = [
    "type", "extension", "size", "sensitivity",
    "Size_sensitivity_Ratio", "Log_Size", "Size_sensitivity_Product",
    "Sqrt_Size", "Size_Squared", "Size_Cubed", "Size_sensitivity_Diff",
    "Size_x_sensitivity_x_Log", "Size_Log_Ratio", "Sensitivity_Squared",
    "Size_Sensitivity_Power", "Is_Large_File", "Is_Small_File", 
    "Is_High_Sensitivity", "Size_Rank", "Sensitivity_Rank"
]

X = df[feature_cols]
y = df["algorithm"]

# Gestion des valeurs infinies et NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# ----------------------------------------------------------
# 5. Scaling robuste
# ----------------------------------------------------------
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# 6. Stratégie d'oversampling améliorée
# ----------------------------------------------------------
categorical_features = [0, 1]  # type et extension

# SMOTEENN combine over et under-sampling
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)

print(f"Taille après resampling: {X_resampled.shape}")
print("Distribution après resampling:")
unique, counts = np.unique(y_resampled, return_counts=True)
for i, count in zip(unique, counts):
    print(f"  Classe {target_encoder.classes_[i]}: {count}")

# ----------------------------------------------------------
# 7. Split stratifié
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.20, random_state=42, stratify=y_resampled
)

X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTailles des splits:")
print(f"Train fit: {X_train_fit.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# ----------------------------------------------------------
# 7b. Validation croisée et recherche d'hyperparamètres
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    'depth': [4, 6],
    'learning_rate': [0.03, 0.05],
    'l2_leaf_reg': [3, 5],
    'iterations': [1000]
}

base_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=0,
    random_seed=42
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print("\nRecherche d'hyperparamètres (peut être long)...")
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Meilleurs paramètres: {best_params}")

# ----------------------------------------------------------
# 8. Configuration CatBoost stabilisée
# ----------------------------------------------------------
def create_stable_catboost(params=None):
    """Configuration CatBoost optimisée pour la stabilité"""

    if params is None:
        params = {}
    
    model = CatBoostClassifier(
        # Paramètres pour la stabilité
        iterations=params.get('iterations', 1200),
        learning_rate=params.get('learning_rate', 0.03),
        depth=params.get('depth', 5),
        l2_leaf_reg=params.get('l2_leaf_reg', 5),
        bagging_temperature=0.8,            # Randomisation contrôlée
        random_strength=0.8,                # Force de randomisation réduite
        
        # Paramètres de robustesse
        bootstrap_type='Bayesian',          # Bootstrap bayésien plus stable
        sampling_frequency='PerTreeLevel',  # Échantillonnage par niveau
        leaf_estimation_method='Newton',    # Méthode d'estimation plus stable
        grow_policy='SymmetricTree',        # Croissance symétrique
        
        # Paramètres spécifiques
        border_count=32,                    # Moins de frontières pour plus de stabilité
        feature_border_type='Median',       # Frontières basées sur la médiane
        
        # Configuration générale
        loss_function="MultiClass",
        eval_metric="Accuracy",
        use_best_model=True,
        od_type='Iter',                     # Early stopping par itération
        od_wait=100,                        # Patience augmentée
        random_seed=42,
        verbose=50
    )
    
    return model

# ----------------------------------------------------------
# 9. Fonctions utilitaires pour analyse
# ----------------------------------------------------------
def plot_stable_learning_curves(model, X_train, X_val, y_train, y_val):
    """Visualisation des courbes d'apprentissage lissées"""
    
    # Entraîner avec logging détaillé
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=100,
        verbose=0
    )
    
    # Récupérer les métriques
    evals = model.get_evals_result()

    # Certaines versions de CatBoost nomment l'ensemble de validation
    # "validation_0" au lieu de "validation". On gère donc les deux cas.
    val_key = 'validation'
    if val_key not in evals:
        # Prendre la première clé autre que "learn" comme validation
        val_key = [k for k in evals if k != 'learn'][0]

    plt.figure(figsize=(15, 5))

    # Original curves
    plt.subplot(1, 3, 1)
    train_scores = evals['learn']['Accuracy']
    val_scores = evals[val_key]['Accuracy']
    
    plt.plot(train_scores, label='Train', alpha=0.7)
    plt.plot(val_scores, label='Validation', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves - Original')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Smoothed curves (moving average)
    plt.subplot(1, 3, 2)
    window = 10
    train_smooth = pd.Series(train_scores).rolling(window=window).mean()
    val_smooth = pd.Series(val_scores).rolling(window=window).mean()
    
    plt.plot(train_smooth, label='Train (lissé)', linewidth=2)
    plt.plot(val_smooth, label='Validation (lissé)', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(f'Smoothed Curves (window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Stability analysis
    plt.subplot(1, 3, 3)
    val_diff = np.diff(val_scores)
    plt.plot(val_diff, alpha=0.7, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Δ Accuracy')
    plt.title('Validation Accuracy Changes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("stable_learning_curves.png")
    plt.close()
    
    # Statistiques de stabilité
    val_stability = np.std(val_scores[-100:])  # Stabilité sur les 100 dernières itérations
    print(f"Stabilité de validation (100 dernières itérations): {val_stability:.6f}")
    
    return val_stability

def analyze_errors(y_true, y_pred, target_encoder):
    """Analyser les erreurs de classification"""
    
    # Matrice de confusion détaillée
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nAnalyse des erreurs par classe:")
    for i, class_name in enumerate(target_encoder.classes_):
        total_actual = np.sum(cm[i, :])
        correct_pred = cm[i, i]
        error_rate = 1 - (correct_pred / total_actual) if total_actual > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  Taux d'erreur: {error_rate:.3f}")
        print(f"  Prédictions correctes: {correct_pred}/{total_actual}")
        
        # Classes les plus confondues
        if total_actual > 0:
            errors = cm[i, :].copy()
            errors[i] = 0  # Enlever les bonnes prédictions
            max_error_idx = np.argmax(errors)
            if errors[max_error_idx] > 0:
                print(f"  Souvent confondu avec: {target_encoder.classes_[max_error_idx]} ({errors[max_error_idx]} fois)")


def plot_learning_curve_ensemble(model, X, y):
    """Tracer la courbe d'apprentissage pour l'ensemble"""

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, "o-", label="Train", color="blue")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="blue",
    )
    plt.plot(train_sizes, val_mean, "o-", label="Validation", color="green")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color="green",
    )
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve_ensemble.eps", format="eps")
    plt.savefig("learning_curve_ensemble.png")
    plt.close()

# ----------------------------------------------------------
# 10. Entraînement du modèle stabilisé
# ----------------------------------------------------------
print("=== ENTRAÎNEMENT CATBOOST STABILISÉ ===")

# Créer le modèle stabilisé avec les meilleurs paramètres de la grille
model = create_stable_catboost(best_params)

print("Entraînement du modèle avec early stopping...")
model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    plot=False
)

# Vérification de stabilité
eval_metrics = model.get_evals_result()
val_key = 'validation'
if val_key not in eval_metrics:
    val_key = [k for k in eval_metrics if k != 'learn'][0]
val_scores = eval_metrics[val_key]['Accuracy']
stability = np.std(val_scores[-100:])
print(f"\nStabilité (écart-type sur 100 dernières itérations): {stability:.6f}")

if stability < 0.005:
    print("✅ Modèle stable")
else:
    print("⚠️ Modèle instable - considérer réduire learning_rate")

# ----------------------------------------------------------
# 11. Ensemble de modèles (optionnel)
# ----------------------------------------------------------
def create_ensemble():
    """Créer un ensemble de modèles"""
    
    # CatBoost pour l'ensemble
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=15,
        l2_leaf_reg=3,
        bootstrap_type='Bayesian',
        loss_function="MultiClass",
        eval_metric="Accuracy",
        use_best_model=False,  # Important: False pour l'ensemble
        random_seed=42,
        verbose=0
    )
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    
    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000, random_state=42, multi_class='ovr'
    )
    
    # Ensemble voting
    ensemble = VotingClassifier([
        ('catboost', cb_model),
        ('random_forest', rf_model),
        ('logistic', lr_model)
    ], voting='soft', n_jobs=1)
    
    print("Entraînement de l'ensemble de modèles...")
    ensemble.fit(X_train, y_train)
    return ensemble

print("\nCréation de l'ensemble de modèles...")
ensemble_model = create_ensemble()

# ----------------------------------------------------------
# 12. Évaluation comparative
# ----------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """Évaluer un modèle"""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
    
    print(f"\n=== {model_name} ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1‑score  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    
    return y_pred

# Évaluation des modèles
y_pred_single = evaluate_model(model, X_test, y_test, "CatBoost Stabilisé")
y_pred_ensemble = evaluate_model(ensemble_model, X_test, y_test, "Ensemble de Modèles")

# Rapport détaillé pour le meilleur modèle
print("\nRapport de classification détaillé (CatBoost Stabilisé):")
print(classification_report(
    y_test, y_pred_single, target_names=target_encoder.classes_, zero_division=0
))

print("\nRapport de classification détaillé (Ensemble de Modèles):")
print(classification_report(
    y_test, y_pred_ensemble, target_names=target_encoder.classes_, zero_division=0
))

# ----------------------------------------------------------
# 13. Visualisations améliorées
# ----------------------------------------------------------
# Importance des features CatBoost
plt.figure(figsize=(6, 4))

feature_importance = model.get_feature_importance()
sorted_idx = np.argsort(feature_importance)[-15:]
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
plt.xlabel('Importance')
plt.title('Top 15 Features - Stable CatBoost')
plt.tight_layout()
plt.savefig('feature_importance_catboost.eps', format='eps')
plt.savefig('feature_importance_catboost.png')
plt.close()

# Matrice de confusion - CatBoost
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_single)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(cmap="Blues", xticks_rotation='horizontal')
plt.title(
    f"Confusion Matrix - Stable CatBoost (Acc: {accuracy_score(y_test, y_pred_single):.3f})"
)
plt.tight_layout()
plt.savefig('confusion_matrix_catboost.eps', format='eps')
plt.savefig('confusion_matrix_catboost.png')
plt.close()

# Matrice de confusion - Ensemble
plt.figure(figsize=(6, 4))
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
disp_ensemble = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=target_encoder.classes_)
disp_ensemble.plot(cmap="Greens", xticks_rotation='horizontal')
plt.title(
    f"Confusion Matrix - Ensemble (Acc: {accuracy_score(y_test, y_pred_ensemble):.3f})"
)
plt.tight_layout()
plt.savefig('confusion_matrix_ensemble.eps', format='eps')
plt.savefig('confusion_matrix_ensemble.png')
plt.close()

# Courbes d'apprentissage - CatBoost
plt.figure(figsize=(6, 4))

eval_metrics = model.get_evals_result()
train_acc = eval_metrics['learn']['Accuracy']
val_key = 'validation'
if val_key not in eval_metrics:
    val_key = [k for k in eval_metrics if k != 'learn'][0]
val_acc = eval_metrics[val_key]['Accuracy']

window = 20
train_smooth = pd.Series(train_acc).rolling(window=window, min_periods=1).mean()
val_smooth = pd.Series(val_acc).rolling(window=window, min_periods=1).mean()
plt.plot(train_acc, alpha=0.3, color='blue', label='Train (original)')
plt.plot(val_acc, alpha=0.3, color='red', label='Validation (original)')
plt.plot(train_smooth, color='blue', linewidth=2, label='Train (lissé)')
plt.plot(val_smooth, color='red', linewidth=2, label='Validation (lissé)')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title("CatBoost Learning Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve_catboost.eps', format='eps')
plt.savefig('learning_curve_catboost.png')
plt.close()

# Courbes d'apprentissage - Ensemble
plot_learning_curve_ensemble(ensemble_model, X_train, y_train)

# Performance comparison
plt.figure(figsize=(6, 4))

models = ['Stable CatBoost', 'Ensemble']
accuracies = [accuracy_score(y_test, y_pred_single), accuracy_score(y_test, y_pred_ensemble)]
colors = ['blue', 'green']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('model_comparison.eps', format='eps')
plt.savefig('model_comparison.png')
plt.close()

# ----------------------------------------------------------
# 14. Analyse détaillée des erreurs
# ----------------------------------------------------------
#analyze_errors(y_test, y_pred_single, target_encoder)

# ----------------------------------------------------------
# 15. Analyse de stabilité approfondie
# ----------------------------------------------------------
print("\n=== ANALYSE DE STABILITÉ ===")
stability_model = create_stable_catboost()
plot_stable_learning_curves(stability_model, X_train_fit, X_val, y_train_fit, y_val)

# ----------------------------------------------------------
# 16. Sauvegarde des modèles
# ----------------------------------------------------------
model.save_model("catboost_stable_model.cbm")
joblib.dump(ensemble_model, "ensemble_stable_model.pkl")
joblib.dump(scaler, "robust_scaler.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n=== MODÈLES SAUVEGARDÉS ===")
print("Fichiers créés:")
print("- catboost_stable_model.cbm")
print("- ensemble_stable_model.pkl")
print("- robust_scaler.pkl")
print("- target_encoder.pkl")
print("- label_encoders.pkl")
print("- feature_importance_catboost.eps")
print("- feature_importance_catboost.png")
print("- confusion_matrix_catboost.eps")
print("- confusion_matrix_catboost.png")
print("- confusion_matrix_ensemble.eps")
print("- confusion_matrix_ensemble.png")
print("- learning_curve_catboost.eps")
print("- learning_curve_catboost.png")
print("- learning_curve_ensemble.eps")
print("- learning_curve_ensemble.png")
print("- model_comparison.eps")
print("- model_comparison.png")

print(f"\n=== RÉSUMÉ FINAL ===")
print(f"Stabilité du modèle: {stability:.6f} {'✅' if stability < 0.005 else '⚠️'}")
print(f"Accuracy CatBoost Stabilisé: {accuracy_score(y_test, y_pred_single):.4f}")
print(f"Accuracy Ensemble: {accuracy_score(y_test, y_pred_ensemble):.4f}")
>>>>>>> daecc8b7a18475378f87b84ba88ecd82ac1a407d
print("Entraînement terminé avec succès!")