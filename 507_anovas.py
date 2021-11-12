#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 18:42:41 2021

@author: coreywiley
"""

## Import packages to be used

import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm, kurtosis, skew, bartlett, levene, shapiro
import matplotlib.pyplot as plt
from sklearn import preprocessing

## Import dataset on stroke prediction

heartFail = pd.read_csv('/Users/coreywiley/Desktop/GRAD_AHI/507/heart_fail.csv')
heartFail_small = heartFail.sample(100)
list(heartFail)

['Age',
 'Sex',
 'ChestPainType',
 'RestingBP',
 'Cholesterol',
 'FastingBS',
 'RestingECG',
 'MaxHR',
 'ExerciseAngina',
 'Oldpeak',
 'ST_Slope',
 'HeartDisease']

## Choose variables to be analyzed, check amount of levels

## DV - Cholestrol (continuous)
## IV 1 - Age (50 levels)
## IV 2 - Resting ECG (3 levels)
## IV 3 - Chest pain type (4 levels)

len(heartFail.Age.value_counts())
heartFail['Age'].value_counts() 

len(heartFail.RestingECG.value_counts()) 
heartFail['RestingECG'].value_counts() 

len(heartFail.ChestPainType.value_counts()) 
heartFail['ChestPainType'].value_counts() 

cholesterol = heartFail['Cholesterol']

## Create new DF with only variables we need

workingDF = heartFail[['Cholesterol','Age','RestingECG','ChestPainType']]

## Assumption Testing (Shapiro)
## DV ~ C(IV) + C(IV)

model = smf.ols("Cholesterol ~ C(Age)", data = workingDF).fit()
shapiro1 = stats.shapiro(model.resid)
## Shapiro 1 (statistic = 0.9325644969940186, pvalue = 6.38509245129749e-20)
## P value is less than 0.05 so there is a significant statistical difference
## between the means, assumption for normality not met

model2 = smf.ols("Cholesterol ~ C(RestingECG)", data = workingDF).fit()
shapiro2 = stats.shapiro(model2.resid)
## Shapiro 2 (statistic = 0.9071518778800964, pvalue = 3.376857628773498e-23)
## P value is less than 0.05 so there is a significant statistical difference
## between the means, assumption for normality not met

model3 = smf.ols("Cholesterol ~ C(ChestPainType)", data = workingDF).fit()
shapiro3 = stats.shapiro(model3.resid)
## Shapiro 3 (statistic = 0.9036797881126404 , pvalue = 1.365252223040331e-23)
## P value is less than 0.05 so there is a significant statistical difference
## between the means, assumption for normality not met

## Boxplots
age_boxplot = sns.boxplot(x='Age', y= 'Cholesterol', data=heartFail, palette="Set3")
restingECG_boxplot = sns.boxplot(x='RestingECG', y= 'Cholesterol', data=heartFail, palette="Set3") 
chestPain_boxplot = sns.boxplot(x='ChestPainType', y= 'Cholesterol', data=heartFail, palette="Set3")

## Barplots
age_barplot = sns.barplot(x='Age', y= 'Cholesterol', data=heartFail, palette="Set2") 
restingECG_barplot = sns.barplot(x='RestingECG', y= 'Cholesterol', data=heartFail, palette="Set2") 
chestPain_barplot = sns.barplot(x='ChestPainType', y= 'Cholesterol', data=heartFail, palette="Set2") 

## Convert data from string to float
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
heartFail = heartFail.apply(le.fit_transform)

heartFail['Age'] = heartFail['Age'].astype('float') 
heartFail['RestingECG'] = heartFail['RestingECG'].astype('float')
heartFail['ChestPainType'] = heartFail['ChestPainType'].astype('float')

##Homogeneity Testing (Levene)

age_Levene = stats.levene(heartFail['Cholesterol'], heartFail['Age'])
## Levene1 - (statistic = 1389.714247315081 , pvalue = 6.711599329876876e-227)
## P value is less than 0.05, unequal variance

ECG_Levene = stats.levene(heartFail['Cholesterol'], heartFail['RestingECG'])
## Levene2 - (statistic = 1973.4738981217413 , pvalue = 3.2046610000653804e-293)
## P value is less than 0.05, unequal variance

CPT_Levene = stats.levene(heartFail['Cholesterol'], heartFail['ChestPainType']) 
## Levene3 - (statistic = 1941.1066553532585 , pvalue = 8.081361691085094e-290)
## P value is less than 0.05, unequal variance


## One way ANOVA Testing - Non-parametric (Kruskal-Wallis)

## Is there a difference between the levels of age and cholesterol?
age_Kruskal = stats.kruskal(heartFail['Cholesterol'], heartFail['Age'])
## Kruskal1 - (statistic = 395.63618206551206 , pvalue = 4.907901566553101e-88)
## P value is less than 0.05, there is a significant statistical difference
## between the levels of age and cholesterol

## Is there a difference between the levels of resting ECG and cholesterol? 
ECG_Kruskal = stats.kruskal(heartFail['Cholesterol'], heartFail['RestingECG'])
## Kruskal2 - (statistic = 625.8040526019138 , pvalue = 4.087015166842414e-138)
## P value is less than 0.05, there is a significant statistical difference
## between the levels of resting ECG and cholesterol

## Is there a difference between the levels of chest pain types and cholesterol? 
CPT_Kruskal = stats.kruskal(heartFail['Cholesterol'], heartFail['ChestPainType'])
## Kruskal3 - (statistic = 761.7121338920518 , pvalue = 1.1396934548105039e-167)
## P value is less than 0.05, there is a significant statistical difference
## between the levels of chest pain types and cholesterol 

## Post-hoc - Howell test
import pingouin as pg

age_ping = pg.pairwise_gameshowell(data=heartFail, dv='Cholesterol', between='Age').round(3)
ECG_ping = pg.pairwise_gameshowell(data=heartFail, dv='Cholesterol', between='RestingECG').round(3)
CPT_ping = pg.pairwise_gameshowell(data=heartFail, dv='Cholesterol', between='ChestPainType').round(3)

heartFail.fillna(0)
heartFail.replace(np.nan, 0)

## Post-hoc keeps giving the following error message : "Cannot convert float 
## NaN to integer"













      
        
        