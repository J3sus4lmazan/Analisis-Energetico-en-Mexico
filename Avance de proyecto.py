#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Conexion a base de datos local
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import geopandas as gpd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


cnx = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password='*****',
    database='miproyecto'
)
cursor = cnx.cursor()


# In[3]:


#Se obtienen la informacion de la tabla generacion bruta por tecnologia
cursor.execute("SELECT * FROM genbrutatech")
GBT = cursor.fetchall() #Generacion bruta por tecnologia
GBTdf = pd.DataFrame(GBT, columns=['GBT_ID', 'Month_Year', 'EnergyTypeID', 'MegaWatt_hourTotal', 'CapMWattPerTech'])
GBTdf = GBTdf.set_index('GBT_ID', drop=True)
GBTdf.head()


# In[4]:


#Se obtienen la informacion de la tabla Meses
cursor.execute("SELECT * FROM months")
M = cursor.fetchall() #Meses
Mdf = pd.DataFrame(M, columns=['Month_YearID', 'MonthDescription'])
Mdf = Mdf.set_index('Month_YearID', drop=True)
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-05','05-Feb','05-Mar','abr-05','05-May','05-Jun','05-Jul',
                 'ago-05','05-Sep','05-Oct','05-Nov','dic-05'],'2005')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-06','06-Feb','06-Mar','abr-06','06-May','06-Jun','06-Jul',
                 'ago-06','06-Sep','06-Oct','06-Nov','dic-06'],'2006')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-07','07-Feb','07-Mar','abr-07','07-May','07-Jun','07-Jul',
                 'ago-07','07-Sep','07-Oct','07-Nov','dic-07'],'2007')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-08','08-Feb','08-Mar','abr-08','08-May','08-Jun','08-Jul',
                 'ago-08','08-Sep','08-Oct','08-Nov','dic-08'],'2008')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-09','09-Feb','09-Mar','abr-09','09-May','09-Jun','09-Jul',
                 'ago-09','09-Sep','09-Oct','09-Nov','dic-09'],'2009')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-10','10-Feb','10-Mar','abr-10','10-May','10-Jun','10-Jul',
                 'ago-10','10-Sep','10-Oct','10-Nov','dic-10'],'2010')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-11','11-Feb','11-Mar','abr-11','11-May','11-Jun','11-Jul',
                 'ago-11','11-Sep','11-Oct','11-Nov','dic-11'],'2011')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-12','12-Feb','12-Mar','abr-12','12-May','12-Jun','12-Jul',
                 'ago-12','12-Sep','12-Oct','12-Nov','dic-12'],'2012')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-13','13-Feb','13-Mar','abr-13','13-May','13-Jun','13-Jul',
                 'ago-13','13-Sep','13-Oct','13-Nov','dic-13'],'2013')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-14','14-Feb','14-Mar','abr-14','14-May','14-Jun','14-Jul',
                 'ago-14','14-Sep','14-Oct','14-Nov','dic-14'],'2014')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-15','15-Feb','15-Mar','abr-15','15-May','15-Jun','15-Jul',
                 'ago-15','15-Sep','15-Oct','15-Nov','dic-15'],'2015')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-16','16-Feb','16-Mar','abr-16','16-May','16-Jun','16-Jul',
                 'ago-16','16-Sep','16-Oct','16-Nov','dic-16'],'2016')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-17','17-Feb','17-Mar','abr-17','17-May','17-Jun','17-Jul',
                 'ago-17','17-Sep','17-Oct','17-Nov','dic-17'],'2017')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-18','18-Feb','18-Mar','abr-18','18-May','18-Jun','18-Jul',
                 'ago-18','18-Sep','18-Oct','18-Nov','dic-18'],'2018')
Mdf['MonthDescription']=Mdf['MonthDescription'].replace(['ene-19','19-Feb','19-Mar','abr-19','19-May','19-Jun','19-Jul',
                 'ago-19','19-Sep','19-Oct','19-Nov','dic-19'],'2019')
Mdf.head()


# In[5]:


#Se obtienen la informacion de tipo de energia
cursor.execute("SELECT * FROM energytypes")
ET = cursor.fetchall() #Tipo de energia
ETdf = pd.DataFrame(ET, columns=['EnergyTypeID', 'EnergyTypeName'])
ETdf = ETdf.set_index('EnergyTypeID', drop=True)
ETdf


# In[6]:


#Se unen las 3 tablas
GBT_full = pd.merge(GBTdf, Mdf, left_on='Month_Year', right_index=True).sort_index()
GBT_full = pd.merge(GBT_full, ETdf, left_on='EnergyTypeID', right_index=True).sort_index()
GBT_full = GBT_full.dropna(axis = 0, how = 'any') #Se retiran los NaN
GBT_full = GBT_full.drop(columns=['Month_Year', 'EnergyTypeID']) #Se retiran las columnas con valores referenciales
GBT_full


# In[7]:


#Se obtiene la informacion de la energia generada por estado
cursor.execute("SELECT GBS_ID, MegaWatt_hourTotal, MonthDescription, StateName FROM genbruperstate JOIN months ON genbruperstate.Month_YearID = months.Month_YearID JOIN states ON states.State_ID = genbruperstate.State_ID;") 
               #genbruperstate.Month_YearID, genbruperstate.State_ID  columnas con valores referenciales
GBS = cursor.fetchall() #Generacion bruta por estado
GBSdf = pd.DataFrame(GBS, columns=['GBS_ID', 'MegaWatt_hourTotal', 'MonthDescription', 'StateName'])
#, 'Month_YearID', 'State_ID'
GBSdf = GBSdf.set_index('GBS_ID', drop=True)
GBSdf = GBSdf.dropna(axis = 0, how = 'any') #Se retiran los NaN
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-05','05-Feb','05-Mar','abr-05','05-May','05-Jun','05-Jul',
                 'ago-05','05-Sep','05-Oct','05-Nov','dic-05'],'2005')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-06','06-Feb','06-Mar','abr-06','06-May','06-Jun','06-Jul',
                 'ago-06','06-Sep','06-Oct','06-Nov','dic-06'],'2006')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-07','07-Feb','07-Mar','abr-07','07-May','07-Jun','07-Jul',
                 'ago-07','07-Sep','07-Oct','07-Nov','dic-07'],'2007')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-08','08-Feb','08-Mar','abr-08','08-May','08-Jun','08-Jul',
                 'ago-08','08-Sep','08-Oct','08-Nov','dic-08'],'2008')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-09','09-Feb','09-Mar','abr-09','09-May','09-Jun','09-Jul',
                 'ago-09','09-Sep','09-Oct','09-Nov','dic-09'],'2009')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-10','10-Feb','10-Mar','abr-10','10-May','10-Jun','10-Jul',
                 'ago-10','10-Sep','10-Oct','10-Nov','dic-10'],'2010')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-11','11-Feb','11-Mar','abr-11','11-May','11-Jun','11-Jul',
                 'ago-11','11-Sep','11-Oct','11-Nov','dic-11'],'2011')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-12','12-Feb','12-Mar','abr-12','12-May','12-Jun','12-Jul',
                 'ago-12','12-Sep','12-Oct','12-Nov','dic-12'],'2012')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-13','13-Feb','13-Mar','abr-13','13-May','13-Jun','13-Jul',
                 'ago-13','13-Sep','13-Oct','13-Nov','dic-13'],'2013')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-14','14-Feb','14-Mar','abr-14','14-May','14-Jun','14-Jul',
                 'ago-14','14-Sep','14-Oct','14-Nov','dic-14'],'2014')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-15','15-Feb','15-Mar','abr-15','15-May','15-Jun','15-Jul',
                 'ago-15','15-Sep','15-Oct','15-Nov','dic-15'],'2015')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-16','16-Feb','16-Mar','abr-16','16-May','16-Jun','16-Jul',
                 'ago-16','16-Sep','16-Oct','16-Nov','dic-16'],'2016')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-17','17-Feb','17-Mar','abr-17','17-May','17-Jun','17-Jul',
                 'ago-17','17-Sep','17-Oct','17-Nov','dic-17'],'2017')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-18','18-Feb','18-Mar','abr-18','18-May','18-Jun','18-Jul',
                 'ago-18','18-Sep','18-Oct','18-Nov','dic-18'],'2018')
GBSdf['MonthDescription']=GBSdf['MonthDescription'].replace(['ene-19','19-Feb','19-Mar','abr-19','19-May','19-Jun','19-Jul',
                 'ago-19','19-Sep','19-Oct','19-Nov','dic-19'],'2019')
GBSdf.head()


# In[8]:


#Se obtiene la informacion de la tabla de capacidad efectiva
cursor.execute("SELECT * FROM effectivecap")
EFC = cursor.fetchall() #Capacidad efectiva
EFCdf = pd.DataFrame(EFC, columns=['EFC_ID', 'MegaWatt', 'Month_YearID', 'State_ID'])
EFCdf = EFCdf.set_index('EFC_ID', drop=True)
#Se obtiene la informacion de la tabla de estados
cursor.execute("SELECT * FROM states")
S = cursor.fetchall() #Estados
Sdf = pd.DataFrame(S, columns=['State_ID', 'StateName'])
Sdf = Sdf.set_index('State_ID', drop=True)
#Se combinan las 3 tablas
EFC_full = pd.merge(EFCdf, Mdf, left_on='Month_YearID', right_index=True).sort_index()
EFC_full = pd.merge(EFC_full, Sdf, left_on='State_ID', right_index=True).sort_index()
EFC_full = EFC_full.dropna(axis = 0, how = 'any') #Se retiran los NaN
EFC_full = EFC_full.drop(columns=['Month_YearID', 'State_ID']) #Se retiran las columnas con valores referenciales
EFC_full


# In[9]:


#Se obtiene la informacion de la tabla de comercio interior
cursor.execute("SELECT * FROM comerciointer")
CI = cursor.fetchall() #Comercio interior
CIdf = pd.DataFrame(CI, columns=['CI_ID', 'MegaWatt_hour', 'QuantityUsers', 'Month_YearID', 'Sector', 'CentsPerKW_hour'])
CIdf = CIdf.set_index('CI_ID', drop=True)
#Se combinan con la tabla de meses
CI_full = pd.merge(CIdf, Mdf, left_on='Month_YearID', right_index=True).sort_index()
CI_full = CI_full.dropna(axis = 0, how = 'any') #Se retiran los NaN
CI_full = CI_full.drop(columns=['Month_YearID']) #Se retiran las columnas con valores referenciales
CI_full.head()


# In[10]:


#Se obtiene la informacion de la tabla de comercio exterior
cursor.execute("SELECT * FROM comercioexter")
CE = cursor.fetchall() #Comercio exterior
CEdf = pd.DataFrame(CE, columns=['CE_ID', 'MegaWatt_hour', 'Month_YearID', 'Country', 'IMP_EXP'])
CEdf = CEdf.set_index('CE_ID', drop=True)
#Se combinan con la tabla de meses
CE_full = pd.merge(CEdf, Mdf, left_on='Month_YearID', right_index=True).sort_index()
CE_full = CE_full.dropna(axis = 0, how = 'any') #Se retiran los NaN
CE_full = CE_full.drop(columns=['Month_YearID']) #Se retiran las columnas con valores referenciales
CE_full.head()


# In[11]:


#¿Cuanta diferencia de energia producida por termoelectica existe con respecto a la fotovoltaica y eolica?
Q1 = GBT_full.groupby('EnergyTypeName').sum()
total_termoelectrica = Q1.iloc[7, 0]
total_fotovoltaica = Q1.iloc[3,0]
total_eolica = Q1.iloc[2,0]
diferencia_1 = total_termoelectrica / total_fotovoltaica
diferencia_2 = total_termoelectrica / total_eolica
print("Se produce", diferencia_1, "veces mas energia termoelectrica que fotovoltaica", "\n", "Se produce", diferencia_2, "veces mas energia termoelectrica que eolica")


# In[12]:


#¿Cuales son los 3 principales estados que más producen energia?
Q2 = GBSdf.groupby('StateName').sum()
Q2 = Q2.sort_values('MegaWatt_hourTotal', ascending=False)
Q2 = Q2.iloc[0:3]
Q2 / 1000000 #cantidad en millon por megawatt hora


# In[13]:


#¿Cual es el estado con mayor capacidad efectiva en la actualidad?
Q3 = EFC_full.groupby('StateName').sum()
Q3 = Q3.sort_values('MegaWatt', ascending=False)
Q3 = Q3.iloc[0:3]
Q3


# In[14]:


#¿Cuál es el sector que más consumo de energía realiza?
Q4 = CI_full.groupby('Sector').sum()
Q4 = Q4.sort_values('MegaWatt_hour', ascending = False)
Q4.iloc[0]


# In[15]:


#¿Cuál es el promedio de energia que se exporta a Estados Unidos?
Q5 = CE_full.groupby('Country').mean()
Q5 = Q5[Q5.index == 'Estados Unidos']
Q5


# In[16]:


#Se cierra el cursor
cursor.close()
#TABLAS
#GBT_full   Generacion bruta por tipo de tecnologia
#GBSdf      Generacion bruta por estado
#CI_full    Comercio Interno
#CE_full    Comercio exterior


# In[17]:


GBT_by_year = GBT_full.groupby('MonthDescription')['MegaWatt_hourTotal'].mean()
Grafica_GBT_1 = GBT_by_year.plot(kind='bar',cmap='cool')
plt.ylabel('Mega Watts hora')
plt.xlabel('Año')
plt.title('Promedio de Energia generada 2005-2019');


# In[18]:


GBT_by_energy = GBT_full.groupby('EnergyTypeName')['MegaWatt_hourTotal'].sum()
plt.figure(figsize=(10,6))
Grafica_2_GBT = GBT_by_energy.plot(kind='pie',cmap='Paired')
plt.ylabel('')
plt.xlabel('Tipos de energia')
plt.title('Energia generada 2005-2019 por tipo de tecnologia');


# In[ ]:


fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True, sharey=True)

sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[1], ax=axes[0, 0])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[2], ax=axes[0, 1])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[3], ax=axes[1, 0])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[4], ax=axes[1, 1])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[5], ax=axes[2, 0])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[6], ax=axes[2, 1])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[7], ax=axes[3, 0])
sns.barplot(GBT_by_energy['MegaWatt_hourTotal'], GBT_by_energy.loc[8], ax=axes[3, 1])

axes[0, 0].set(xlabel='', ylabel='', title='Rango de Precio: 1')
axes[0, 1].set(xlabel='', ylabel='', title='Rango de Precio: 2')
axes[1, 0].set(xlabel='', ylabel='', title='Rango de Precio: 3')
axes[1, 1].set(xlabel='', ylabel='', title='Rango de Precio: 4')
axes[2, 0].set(xlabel='', ylabel='', title='Rango de Precio: 5')
axes[2, 1].set(xlabel='', ylabel='', title='Rango de Precio: 6')
axes[3, 0].set(xlabel='', ylabel='', title='Rango de Precio: 7')
axes[3, 1].set(xlabel='', ylabel='', title='Rango de Precio: 8')

fig.suptitle('Energia generada separada por tipo', fontsize=15);


# In[19]:


N = GBT_full.groupby('MonthDescription').sum()
sns.set(style="whitegrid");

sns.boxplot(x=N['MegaWatt_hourTotal']);
plt.axvline(x=N['MegaWatt_hourTotal'].mean(), c='y');

iqr = N['MegaWatt_hourTotal'].quantile(0.75) - N['MegaWatt_hourTotal'].quantile(0.25)
filtro_inferior = N['MegaWatt_hourTotal'] > N['MegaWatt_hourTotal'].quantile(0.25) - (iqr * 1.5) 
filtro_superior = N['MegaWatt_hourTotal'] < N['MegaWatt_hourTotal'].quantile(0.75) + (iqr * 1.5)

df_filtrado = N[filtro_inferior & filtro_superior]
sns.boxplot(df_filtrado['MegaWatt_hourTotal']);


# In[ ]:


import geopandas as gpd
df_geo = gpd.read_file('../Procesamiendo de datos/mexico.json')
df_geo = df_geo.merge(EFC_full, left_on='name', right_on='StateName')
geojson = '../Procesamiendo de datos/mexico.json'
df_geo.head(2)


# In[ ]:


import folium
mapa = folium.Map(location=[24.5, -99], zoom_start=4.5)

## LAYER 1 - Mapa cloropletico
folium.Choropleth(
    geo_data=geojson,
    data=EFC_full,
    columns=['StateName', 'MegaWatt'],
    key_on='feature.properties.name',
    fill_color='YlOrBr',
).add_to(mapa)


## Layer 2 - Tooltip

## CSS
style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

tooltip = folium.features.GeoJson(
    df_geo,
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['StateName', 'MegaWatt'],
        aliases=['Estado', 'Mega Watts Generados']
    )
)

mapa.add_child(tooltip)
mapa.keep_in_front(tooltip)
folium.LayerControl().add_to(mapa)

mapa


# In[20]:


#Machine Learning
#Aqui elegiremos 2 campos: Cantidad de energia consumida, y cantidad de usuarios. Dando como resultado el tipo de sector
X = CI_full[['MegaWatt_hour', 'QuantityUsers']]
Y = CI_full['Sector'].map({
    'Residencial':1,
    'Comercial':2,
    'Servicios':3,
    'Agricola':4,
    'Empresa Mediana':5,
    'Gran Industria':6})


# In[23]:


CI_full


# In[21]:


#Dividir los datos en Train y Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[22]:


#Entrenamiento y prediccion
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[ ]:


y_train_predict = lin_model.predict(X_train)
MSE = mean_squared_error(Y_train,y_train_predict)
print("Entrenamiento: MSE ="+str(MSE))

y_test_predict = lin_model.predict(X_test)
MSE = (mean_squared_error(Y_test, y_test_predict))
print("Pruebas: MSE ="+str(MSE))


# In[ ]:


#Comparacion de predicciones
df_predicciones = pd.DataFrame({'valor_real':Y_test, 'prediccion':y_test_predict})
df_predicciones = df_predicciones.reset_index(drop = True)
df_predicciones.head(10)


# In[ ]:


#Evaluar el modelo
from sklearn.svm import SVC #Support Vector Classifier
from sklearn.pipeline import make_pipeline


# In[ ]:


#Evaluar SVM
SupportVectorMachine = SVC()
SupportVectorMachine.fit(x_train, y_train) 
y_pred_svm = SupportVectorMachine.predict(x_test) 


# In[ ]:


evaluar(y_test, y_pred_svm)


# In[ ]:


#Matriz de confusion
from sklearn.metrics import confusion_matrix

def calcularAccuracy(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100
    return accuracy
def calcularSensibilidad(TP, TN, FP, FN):
    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100
    return sensibilidad
def calcularEspecificidad(TP, TN, FP, FN):
    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    return especificidad

resultado = confusion_matrix(Y_test, y_test_predict)
print(resultado)
(TN, FP, FN, TP) = resultado.ravel()
print("True positives: "+str(TP))
print("True negatives: "+str(TN))
print("False positives: "+str(FP))
print("False negative: "+str(FN))

acc = calcularAccuracy(TP, TN, FP, FN)
sen = calcularSensibilidad(TP, TN, FP, FN)
spec = calcularEspecificidad(TP, TN, FP, FN)
print("Precision:"+str(acc)+"%")
print("Sensibilidad:"+str(sen)+"%")
print("Especificidad:"+str(spec)+"%")


# In[24]:


#Serie de tiempo de la generacion de energia por tipo
#transformacion del timeserie
series = GBT_full['MegaWatt_hourTotal'].to_numpy()


# In[25]:


split_time = 1000

x_train = series[:split_time]
x_test = series[split_time:]


# In[26]:


#Se definen los datos de entrada y de salida seguna la ventana de tiempo definida
window_size = 20

X = None 
Y = None

for counter in range(len(x_train)-window_size-1):
  muestra = np.array([x_train[counter:counter+window_size]])
  salida = np.array([x_train[counter+window_size]])
  if X is None:
    X = muestra
  else:
    X = np.append(X,muestra,axis=0)
  if Y is None:
    Y = salida
  else:
    Y = np.append(Y,salida)


# In[27]:


#Se generan las capas 
l0 = tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu")
l1 = tf.keras.layers.Dense(10, activation="relu")
l_output = tf.keras.layers.Dense(1)


# In[28]:


model = tf.keras.models.Sequential([l0,l1,l_output])


# In[29]:


model.compile(loss="mse",optimizer=tf.keras.optimizers.SGD(lr=1e-6,momentum=0.9),metrics=['mae'])


# In[30]:


#Se entrena el modelo
model.fit(X,Y,epochs=100,batch_size=32,verbose=1,validation_split=0.2)


# In[31]:


#Se definen las predicciones
forecast = []
for time in range(len(series)-window_size):
  forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]


# In[32]:


results = np.array(forecast)[:,0,0]


# In[33]:


#Grafica
plt.figure(figsize=(10,6))
plt.plot(x_test,"-")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)

plt.plot(results,"-")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)


# In[ ]:




