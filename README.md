# P7_OpenClassrooms
Repository pour P7 - Parcours Data Science - OpenClassrooms
Projet 7 du Parcours Data Science d'OpenClassrooms.

Lien pour télécharger les datasets : https://www.kaggle.com/c/home-credit-default-risk/data

Objectif du projet :
 - Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
 - Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions
   faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

On commence par choisir un kernel Kaggle pour pouvoir se concentrer sur l'élaboration du modèle et son optimisation.

Kernel choisi : https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
Problème : il reste des valeurs manquantes (car utilisation de lightgbm comme moteur de classification).

Nous en profitons pour supprimer des variables peu significatives en plus de gérer les dernières valeurs manquantes.
