
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import time
import seaborn as sns
import matplotlib.pyplot as plt



# Chargement des données à partir du fichier CSV
data = pd.read_csv("data.csv")

# Compter les occurrences de chaque classe
class_counts = data['diagnosis'].value_counts()

# Visualisation de la distribution des classes
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution des classes')
plt.show()


# Encodage des étiquettes de classe
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Séparation des caractéristiques et des étiquettes de classe
X = data.drop('diagnosis', axis=1)  # Caractéristiques
y = data['diagnosis']  # Étiquettes de classe

# Division du jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation des valeurs manquantes avec la stratégie 'most_frequent'
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Dictionnaire pour stocker les performances des différents noyaux
performance = {}

# Noyaux à tester
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    start_time = time.time()  # Mesurer le temps de début

    # Création et ajustement du modèle SVM

    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train_scaled, y_train)

    # Prédiction sur l'ensemble de test

    y_pred_svm = svm_model.predict(X_test_scaled)

    # Calcul de l'exactitude (accuracy)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    # Calcul de la précision
    precision_svm = precision_score(y_test, y_pred_svm)

    # Calcul du rappel
    recall_svm = recall_score(y_test, y_pred_svm)

    # Calcul du F1-score
    f1_svm = f1_score(y_test, y_pred_svm)

    # Stocker les performances dans le dictionnaire
    performance[kernel] = {'Accuracy': accuracy_svm, 'Precision': precision_svm, 'Recall': recall_svm, 'F1-score': f1_svm}


# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Afficher la matrice de confusion
print("Matrice de confusion :")
print(conf_matrix)

# Tracer la matrice de confusion avec seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)

# Ajouter des étiquettes et un titre
plt.xlabel('Classe Prédite')
plt.ylabel('Classe Réelle')
plt.title('Matrice de Confusion')
plt.show()


# Création d'un DataFrame à partir du dictionnaire de performances
df_performance = pd.DataFrame.from_dict(performance, orient='index')

# Affichage du tableau de performances
print("\nTableau de performances :")
print(df_performance)






