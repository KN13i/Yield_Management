# -*- coding: utf-8 -*-
#Importation des packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

#Lecture des données
file = "C:/Users/killi/Documents/M1 S2/Mémoire/algorithm price dynamic/dynamic_pricing.csv"
df = pd.read_csv(file)
df.head()

#Nombre de valeurs manquantes par variable
df.isna().sum() #Pas de valeurs manquantes
df.info

#Changer les variables qualitatives en variable quantitative
df.Location_Category = df.Location_Category.map({
    "Urban":0,
    "Rural":1,
    "Suburban":2})

df.Customer_Loyalty_Status = df.Customer_Loyalty_Status.map({
    "Regular":0,
    "Silver":1,
    "Gold":2})

df.Time_of_Booking = df.Time_of_Booking.map({
    "Morning":0,
    "Afternoon":1,
    "Evening":2,
    "Night":3})

df.Vehicle_Type = df.Vehicle_Type.map({
    "Premium":0,
    "Economy":1})

#Visualisation des varaibles
tab = df.describe()
sns.pairplot(df)

#Graphique en barres
sns.histplot(df['Time_of_Booking'], bins=4)  # Utilisation de bins pour spécifier le nombre de bacs ou de barres
plt.title('Réservation selon les périodes de la journée ')
plt.xlabel('Moment de la journée')
plt.xticks([0, 1, 2,3],labels=["Morning","Afternoon","Evening", "Night"])
plt.ylabel('Nombre de réservations')
plt.show()

#Boîte à moustache 
sns.boxplot(data=df, x='Time_of_Booking', y='Historical_Cost_of_Ride', palette='GnBu')
plt.xlabel('Moment de la journée')
plt.xticks([0, 1, 2,3],labels=["Morning","Afternoon","Evening", "Night"])
plt.ylabel('Prix des réservations')
plt.show()

#Nuage de points
sns.regplot(x='Expected_Ride_Duration', y='Historical_Cost_of_Ride', data=df, 
            scatter=True, color='cornflowerblue', line_kws={"color": "green"})
plt.xlabel('Durée de la course')
plt.ylabel('Prix des trajets')
plt.show()

#Boîte à moustache avec toutes les variables qualitatives recodées
cat = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
plt.figure(figsize=(12,10))
for i, c in enumerate(cat, 1):
    plt.subplot(2,2,i)
    sns.boxplot(y=df['Historical_Cost_of_Ride'], x=df[c],  palette='GnBu')    
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

#Matrice de corrélation
matrice_correlation = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(matrice_correlation, annot=True, cmap='crest', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


#Modèle de régression linéaire
# Séparation des variables
X = df.drop("Historical_Cost_of_Ride", axis=1) #on supprime la variable sur le prix, on a seulement les varaibles explicatives
Y = df["Historical_Cost_of_Ride"] #Juste la variable à expliquer
#Division des données entre le test (20%) et l'entraînement du modèle (80%) 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1) 
# Modèle
model = LinearRegression()
model.fit(X_train, Y_train)
R2 = model.score(X_test, Y_test)
print (R2) #R2 = 0.90
#La régression linéaire explique efficacement la prédiction avec l'ensemble des données

#Coefficients de chaque variables explicatives
coeff = model.coef_
intercept = model.intercept_
predictions = model.predict(X_test)

#Visualiser régression multiple
plt.scatter(Y_test, predictions, color='blue')
plt.plot(Y_test, Y_test, color='red', linewidth=2)  # Ligne de référence : valeurs réelles = valeurs prédites
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Régression multiple : Valeurs prédites vs Valeurs réelles')
plt.show()

#Prédiction
exemple_trajet = [80,10,1,2,20,4,3,1,120]
estimation = sum(coeff*exemple_trajet)+ intercept
print (estimation) # Prix de 407.76
#D'après les données mis en exemple, la valeur prédite serait de 407,76 euros.

# Classification par clusters
# Créer une liste pour stocker les valeurs de distortion
distortions = []

# Tester différentes valeurs de k (nombre de clusters) pour trouver le coude
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Tracer le graphique du coude
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Méthode du Coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('Distortion')
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Extraire les centres de chaque cluster
centers = kmeans.cluster_centers_

# Tracer le diagramme de dispersion avec les clusters colorés
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='red', s=200, edgecolor='k')
plt.title('Visualisation des clusters générés par K-means')
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.show()
