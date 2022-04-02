import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans

"""
comentariu multiline
sau ctrl + /
"""


#   Citirea datelor

raw_data = pd.read_csv('audi.csv')
data = raw_data

#   Verificarea valorilor lipsa

print("Verificarea valorilor lipsa:")
print(data.isnull().sum())

#   Vizualizarea datelor pentru identificarea outlierilor

# print(data.head())
data.describe(include = 'all').to_csv("data describe.csv")

#   CURATARE

#   Eliminarea outlierilor (primii 1%) pentru variabilele 'price', 'mileage' si 'mpg'

q_price = data['price'].quantile(0.99)
q_mileage = data['mileage'].quantile(0.99)
q_mpg = data['mpg'].quantile(0.99)
q_engineSize = data['engineSize'].quantile(0.99)

data_1 = data[data['price'] < q_price]
data_2 = data_1[data_1['mileage'] < q_mileage]
data_3 = data_2[data_2['mpg'] < q_mpg]
data_4 = data_3[data_3['engineSize'] > 0]
data_5 = data_4[data_4['engineSize'] < 5]

date_curatate = data_5

#   Vizualizarea datelor curatate

date_curatate.describe(include='all').to_csv("data describe 2.csv")

#   MULTICOLINIARITATE

#   Folosirea indicatorul VIF pentru a vedea daca exista un grad mare de multicoliniaritate

variabile = date_curatate[['mileage', 'year', 'mpg', 'engineSize']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variabile.values, i) for i in range(variabile.shape[1])]
vif["Variabila"] = variabile.columns

print("Valorile vif initiale:")
print(vif)

#   Eliminarea variabilei 'year' deoarece prezinta cea mai mare valoare VIF

date_no_mc = date_curatate.drop(['year'], axis=1)

variabile_1 = date_no_mc[['mileage', 'mpg', 'engineSize']]
vif1 = pd.DataFrame()
vif1["VIF"] = [variance_inflation_factor(variabile_1.values, j) for j in range(variabile_1.shape[1])]
vif1["Variabila"] = variabile_1.columns

#   Micsorarea considerabila a valorilor VIF dupa eliminarea variabilei 'year'

print("Valorile vif dupa eliminarea variabilei 'year':")
print(vif1)

#   Salvarea datelor folosite in regresie

date_no_mc.to_csv('Date regresie.csv', index=False)

"""

#   REGRESIE

#   Declararea variabilelor

y = date_no_mc['price']
x1 = date_no_mc[['mileage', 'mpg', 'engineSize']]

#   Crearea regresiei

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()

#   Rezultatele regresiei

print("Rezultatele regresiei:")
print(results.summary())

#   Matricea de corelatie

print("Matricea de corelatie:")
date_corelatie = date_no_mc.drop(['tax'], axis=1)
print(date_corelatie.corr())

"""

"""

#   GRAFICE

#   Distributia variabilei 'price' inainte de curatarea outlierilor

plt.hist(data['price'], bins=15)
plt.title("Distributia variabilei 'price' inainte de curatare")
plt.show()

#   Distributia variabilei 'price' dupa curatarea outlierilor


plt.hist(date_no_mc['price'], bins=15)
plt.title("Distributia variabilei 'price' dupa curatare")
plt.show()

#   Distributia variabilei 'price' dupa centrare si standardizare

date_cs = date_no_mc
date_cs['price_cs'] = (date_cs['price'] - np.mean(date_cs['price'])) / np.std(date_cs['price'])
plt.hist(date_cs['price_cs'], bins=15)
plt.title("Distributia variabilei 'price' dupa curatare")
plt.show()

#   Scatter plot 1 (logaritmare)


date_log = date_no_mc
date_log['price_log'] = np.log(date_log['price'])
date_log['mileage_log'] = np.log(date_log['mileage'])
plt.scatter(date_log['mileage_log'], date_log['price_log'])
dx = date_log['mileage_log']
dy = date_log['price_log']
m, b = np.polyfit(dx, dy, 1)
plt.plot(dx, m * dx + b, color='red')
plt.title("Corelatie price - mileage")
plt.xlabel("mileage_log")
plt.ylabel("price_log")
plt.show()

#   Scatter plot 1

plt.scatter(date_no_mc['mileage'], date_no_mc['price'])
dx = date_no_mc['mileage']
dy = date_no_mc['price']
m, b = np.polyfit(dx, dy, 1)
plt.plot(dx, m * dx + b, color='red')
plt.title("Corelatie price - mileage")
plt.xlabel("mileage")
plt.ylabel("price")
plt.show()


#   Scatter plot 2

plt.scatter(date_no_mc['mpg'], date_no_mc['price'])
dx = date_no_mc['mpg']
dy = date_no_mc['price']
m, b = np.polyfit(dx, dy, 1)
plt.plot(dx, m * dx + b, color='red')
plt.title("Corelatie price - mpg")
plt.xlabel("mpg")
plt.ylabel("price")
#plt.show()

#   Scatter plot 3

plt.scatter(date_no_mc['engineSize'], date_no_mc['price'])
dx = date_no_mc['engineSize']
dy = date_no_mc['price']
m, b = np.polyfit(dx, dy, 1)
plt.plot(dx, m * dx + b, color='red')
plt.title("Corelatie price - engineSize")
plt.xlabel("engineSize")
plt.ylabel("price")
plt.show()

#   CALCULE

#   Pret si consum - medii pe modele

date_grupate_Medii = date_curatate.groupby(['model']).agg({'price': ['mean'], 'mpg': ['mean'], 'model': ['count']})
date_grupate_Medii.columns = ['Mean price', 'Mean mpg', 'Count model']
date_grupate_Medii = date_grupate_Medii.reset_index()
date_grupate_Medii.to_csv('Medii.csv', index=False)

#   Transmisie, combustibil si motorizare - preferate


date_grupate_Preferinte = date_curatate.groupby(['model'])[['transmission', 'fuelType', 'engineSize']].agg(
    pd.Series.mode)
date_grupate_Preferinte.columns = ['Mode model', 'Mode fuelType', 'Mode engineSize']
date_grupate_Preferinte = date_grupate_Preferinte.reset_index()
date_grupate_Preferinte.to_csv('Preferinte.csv', index=False)

"""

#   CLUSTERIZARE

#   Alegerea numarului de clusteri

#   WCSS (minimize the within-cluster sum of squares)

from sklearn import preprocessing
c = date_no_mc[['price', 'engineSize']]
c_scaled = preprocessing.scale(c)
print(c_scaled)

wcss = []
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(c_scaled)
    wcss.append(kmeans.inertia_)

#   The Elbow Method

# plt.plot(range(1,10), wcss)
# plt.xlabel('Numar clusteri')
# plt.ylabel('WCSS')
# plt.show()

#   Clusterizare

#   Calcularea clusterului

kmeans = KMeans(3)
kmeans.fit(c_scaled)
clusters = kmeans.fit_predict(c_scaled)
# print(clusters)

#   Atribuire clusteri

data_with_clusters_scaled = c
data_with_clusters_scaled['cluster'] = clusters
print(data_with_clusters_scaled)

plt.scatter(data_with_clusters_scaled['engineSize'], data_with_clusters_scaled['price'], c=data_with_clusters_scaled['cluster'], cmap='rainbow')
plt.xlabel('egineSize')
plt.ylabel('price')
plt.show()

