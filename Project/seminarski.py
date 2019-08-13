import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
import sklearn.preprocessing as prep

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import  DictVectorizer
from sklearn.tree import  DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import  GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import  KNeighborsClassifier

import sklearn.metrics as met
from sklearn.feature_extraction.text import TfidfTransformer

import subprocess

#ucitavanje dve tabele, njihovo spajanje, biranje znacajnih atributa
demographic = pd.read_csv('demographic.csv')
happy = pd.read_csv('cleaned_hm.csv')
happy = pd.merge(happy, demographic, on='wid', validate='m:1')
cols = ['parenthood', 'gender', 'age', 'country', 'cleaned_hm', 'reflection_period', 'predicted_category', 'marital']
happy = happy[cols]

#24h -> 0
#3m -> 1
happy.loc[happy['reflection_period'] == '24h', 'reflection_period'] = 0
happy.loc[happy['reflection_period'] == '3m', 'reflection_period'] = 1

happy.rename(columns = {'3m' : '3m or 24h'}, inplace = True)

#i izbacujemo redove gde nedostaje bracni status
happy.loc[happy['marital']=='single', 'marital']=0
happy.loc[happy['marital']=='married', 'marital']=1
happy.loc[happy['marital']=='separated', 'marital']=2
happy.loc[happy['marital']=='divorced', 'marital']=3
happy.loc[happy['marital']=='widowed', 'marital']=4

happy['marital'].dropna(inplace=True)

#m -> 0
#z -> 1
#drugi -> 2
happy.loc[happy['gender']=='m', 'gender']=0
happy.loc[happy['gender']=='f', 'gender']=1
happy.loc[happy['gender']=='o', 'gender']=2

happy['gender'] = pd.to_numeric(happy['gender'], errors='coerce')
happy['gender'].dropna(inplace=True)

#ima 7 predicted category pa cemo svakoj da dodelimo jedan broj
print(pd.pivot_table(happy, index='predicted_category', aggfunc='count'))

"""
happy.loc[happy['predicted_category']=='exercise', 'predicted_category']=1
happy.loc[happy['predicted_category']=='enjoy_the_moment', 'predicted_category']=2
happy.loc[happy['predicted_category']=='achievement', 'predicted_category']=3
happy.loc[happy['predicted_category']=='nature', 'predicted_category']=4
happy.loc[happy['predicted_category']=='bonding', 'predicted_category']=5
happy.loc[happy['predicted_category']=='affection', 'predicted_category']=6
happy.loc[happy['predicted_category']=='leisure', 'predicted_category']=7
happy['predicted_category'] = pd.to_numeric(happy['predicted_category'], errors='coerce').astype('float64')
happy['predicted_category'].dropna(inplace=True)

"""
#parrenthood no -> 0
#            yes -> 1
happy.loc[happy['parenthood']=='n', 'parenthood']=0
happy.loc[happy['parenthood']=='y', 'parenthood']=1
happy['parenthood'] = pd.to_numeric(happy['parenthood'], errors='coerce')
happy['parenthood'].dropna(inplace=True)

#upisujemo nove podatke u 'izlaz.csv'
#print(happy.head(10))
happy.to_csv('izlaz.csv')

#ucitavanje podataka nakon pretprocesiranja
df = pd.read_csv('izlaz.csv')
df = df.replace(np.inf, np.nan)
df = df.replace(-np.inf, np.nan)
df = df.dropna()

print(df.info())
print(df.describe())

#funkcija za ispis informacija o primenjenoj metodi klasifikacije
def class_info(clf_arg, x_train_arg, y_train_arg, x_test_arg, y_test_arg):

    clf_arg.fit(x_train_arg, y_train_arg)
    #distances, indices = clf.kneighbors(x_test_arg)
    #print('distances', distances)
    #print('indices', indices)

    y_pred = clf_arg.predict(x_test_arg)

    cnf_matrix = met.confusion_matrix(y_test_arg, y_pred)
    df_cnf_matrix = pd.DataFrame(cnf_matrix, index=clf.classes_, columns=clf.classes_)
    print(df_cnf_matrix)
    #print("Matrica konfuzije", cnf_matrix, sep="\n")
    print("\n")

    accuracy = met.accuracy_score(y_test_arg, y_pred)
    print("Preciznost", accuracy)
    print("\n")

    class_report = met.classification_report(y_test_arg, y_pred)
    #print("Izvestaj klasifikacije", class_report, sep="\n")

#TODO treba dodati i za godine
#TODO vizualizacija drveta
#prvo cemo klasifikaciju vrsiti na osnovu atributa: 'parenthood', 'gender', 'marital'

features = ["parenthood", "gender", "marital"]
x_original = df[features]

#normalizacija
x = pd.DataFrame(prep.MinMaxScaler().fit_transform(x_original))
x = x[:10000]
print(x)
x.columns = features
y = df["predicted_category"]
y = y[:10000]



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify = y)

"""
#KNN
k_values = range(3,10)
p_values = [1, 2]
weights_values = ['uniform', 'distance']

for k in k_values:
    for p in p_values:
        for weight in weights_values:
            clf = KNeighborsClassifier(n_neighbors=k,
                                        p=p,
                                        weights=weight)

            print(colored("k="+ str(k), "blue"))
            print(colored("p="+str(p), "blue"))
            print(colored("weight=" + weight, "blue") )

            class_info(clf, x_train, y_train, x_test, y_test)

#DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=2)
class_info(clf, x_train, y_train, x_test, y_test)

"""

