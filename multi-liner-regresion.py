import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as AS

data_file = pd.read_csv("c116/data-logic.csv")

data_file_EstimatedSalary = data_file["EstimatedSalary"].tolist()
data_file_Purchased =  data_file["Purchased"].tolist()
data_file_Age = data_file["Age"].tolist()


data_file_plot_graph  = px.scatter(x = data_file_Age , y =  data_file_Purchased )
# data_file_plot_graph.show()

colors = []

for i in data_file_Purchased:
    if i ==1 :
        colors.append("green")
    else:
        colors.append("red")
        
        
color_graph =  go.Figure(data = go.Scatter(x = data_file_EstimatedSalary , y = data_file_Age ,mode =  "markers" , marker = dict(color = colors)))
# color_graph.show()

factors = data_file[["EstimatedSalary","Age"]]
outcome = data_file["Purchased"]

salary_train , salary_test , Purchased_train , Purchased_test =  tts(factors,outcome,test_size = 0.25 ,random_state=0)

sc = StandardScaler()

salary_train = sc.fit_transform(salary_train)
salary_test = sc.transform(salary_test)

lr = LogisticRegression(random_state = 0 )

purchased_prediction = lr.fit(salary_test,Purchased_train)

result  = AS(Purchased_test , purchased_prediction)
print(result)

