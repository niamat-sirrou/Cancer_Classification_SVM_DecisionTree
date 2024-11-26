import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc


# Lire le fichier CSV avec pandas
dataset = pd.read_csv('data.csv')

# Afficher les premières lignes du DataFrame
print(dataset.head())

# Afficher des statistiques descriptives sur les données
print(dataset.describe())

#le nombre de lignes et le nombre de colonnes du Dataframe.
print(dataset.shape)

# Obtenir des informations sur le Dataframe
dataset.info()

#visualiser la distribution des classes dans la colonne 'diagnosis' du Dataframe
plot = dataset['diagnosis'].value_counts().plot(kind='bar', title="Class distributions : Benign | Malignant")
fig = plot.get_figure()

# Vérifier les valeurs manquantes (NaN) et compter le nombre dans chaque colonne
valeurs_manquantes_par_colonne = dataset.isna().sum()
print("valeurs_manquantes_par_colonne")
print(valeurs_manquantes_par_colonne)

#Vérifiez les lignes en double
valeurs_dupliquées = dataset.duplicated().sum()
print("valeurs_dupliquées",valeurs_dupliquées)


# split data
x = dataset.drop(["diagnosis"] , axis = 1)
y = dataset["diagnosis"]

# split train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Initialiser le modèle de l'arbre de décision
model = DecisionTreeClassifier(random_state=42)

# Entraîner le modèle sur les données d'entraînement
model.fit(x_train, y_train)

# Visualiser l'arbre de décision
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=x.columns, class_names=y.unique(), filled=True)
plt.show()

# Prédire les étiquettes sur les données de test
y_pred = model.predict(x_test)

# Convertir les étiquettes en valeurs binaires pour la classe positive ('M')
y_test_binary = [1 if label == 'M' else 0 for label in y_test]
y_pred_binary = [1 if label == 'M' else 0 for label in y_pred]


# Calculer l'accuracy 
accuracy = accuracy_score(y_test, y_pred)

# Calculer la précision
precision = precision_score(y_test, y_pred, pos_label='M', average='binary')

# Calculer le rappel
recall = recall_score(y_test, y_pred, pos_label='M', average='binary')

# Calculer le F1-score
f1 = f1_score(y_test, y_pred, pos_label='M', average='binary')

# Afficher les résultats
print("Accuracy (classe 'M') :", accuracy)
print("Précision (classe 'M') :", precision)
print("Rappel (classe 'M') :", recall)
print("F1-score (classe 'M') :", f1)


# Création de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("matrice de confusion")
print(cm)

# Visualiser la matrice de confusion avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Matrice de confusion')
plt.show()


# Calculer la courbe ROC
fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)

# Calculer l'AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Afficher les valeurs
print("Taux de faux positifs (FPR) :", fpr)
print("Taux de vrais positifs (TPR) :", tpr)
print("AUC (Area Under the Curve) :", roc_auc)

# Tracer la courbe ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()

