# Classification du Cancer du Sein : Analyse Comparative des Algorithmes SVM et Arbre de Décision

## Description du Projet
Ce projet vise à effectuer une analyse comparative entre deux algorithmes de classification populaires : **SVM (Support Vector Machine)** et **Arbre de Décision**. L'objectif est de classer les tumeurs du sein en **malignes** ou **bénignes** en se basant sur des caractéristiques extraites d'images numériques.  
Les étapes principales incluent le prétraitement des données, l'entraînement des modèles, l'évaluation des performances et l'interprétation des résultats.

---

## Table des Matières
- [Données](#données)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Étapes du Projet](#étapes-du-projet)
  - [Prétraitement des Données](#prétraitement-des-données)
  - [Modèle SVM](#modèle-svm)
  - [Modèle Arbre de Décision](#modèle-arbre-de-décision)
  - [Évaluation des Modèles](#évaluation-des-modèles)
- [Résultats](#résultats)
- [Conclusion](#conclusion)
- [Licence](#licence)

---

## Données
Les données utilisées dans ce projet contiennent des informations sur les tumeurs du sein, notamment des caractéristiques telles que :
- Rayon moyen
- Texture moyenne
- Périmètre moyen
- Etc.  

Ces données ont été nettoyées, normalisées et prêtes à l'emploi.

---

## Prérequis
Assurez-vous d'avoir installé les dépendances suivantes avant de commencer :
- Python 3.x
- Bibliothèques : 
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

---

## Installation
Clonez ce repository et installez les dépendances nécessaires :  
```bash
git clone https://github.com/niamat-sirrou/Cancer_Classification_SVM_DecisionTree.git
cd Cancer_Classification_SVM_DecisionTree
pip install -r requirements.txt
