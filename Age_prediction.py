#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 01:12:59 2017

@author: shamim
"""
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def trainClassifier(file):
    df = pd.read_csv(file)
    df["Sex"]=df["Sex"].replace({"male":0})
    df["Sex"]=df["Sex"].replace({"female":1})
    
    df["Embarked"]=df["Embarked"].replace({"S":0})
    df["Embarked"]=df["Embarked"].replace({"C":1})
    df["Embarked"]=df["Embarked"].replace({"Q":2})
    df = df.drop(["Name","Ticket","Fare","Cabin"],axis=1)
    df = df.dropna(subset=["Age"])
    
    X = df.loc[:,["Pclass","Sex","SibSp","Parch","Embarked"]]
    drop = X.dropna(subset=["Embarked"])
    X.loc[:,"Embarked"]=X.loc[:,"Embarked"].fillna(drop['Embarked'].value_counts().max())
    
    for index,row in df.iterrows():
        if row["Age"] <= 16:
            df.loc[index,"Age"] = 1
        elif row["Age"]<=40:
            df.loc[index,"Age"] = 2
        elif row["Age"] <=55:
            df.loc[index,"Age"] = 3
        else:
            df.loc[index,"Age"] = 4
    y = df.loc[:,"Age"]
#    print(y)
#    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8,stratify=y)
#    print(X_train.head())
#    print(X_test.head())
#    print(y_train.head())
#    print(y_test.head())
    clf = LogisticRegression()
    
    clf = clf.fit(X,y)
    return clf
    
#    s = clf.score(X_test,y_test)
#    print(s)
    
    
#if __name__ =="__main__":
#    clf = trainClassifier("train.csv")
#    s=clf.predict([0,1,1,2,1])
#    print(s)
    
    
    
    