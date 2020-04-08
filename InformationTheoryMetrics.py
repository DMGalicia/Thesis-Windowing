from os import scandir, getcwd
from scipy.io import arff
from time import time
import pandas as pd
import numpy as np
import math

def probDist(n1):
    #Where n1 is the path to a ARFF file
    #It returns a string list with the probability distribution of a dataset
    data1 = arff.loadarff(n1)
    df1 = pd.DataFrame(data1[0])
    headers = df1.columns
    classID = len(df1.columns)-1
    classes = df1[headers[classID]].unique()
    instances = len(df1)
    distribution = []
    j=0
    classes.sort(axis=0)
    for i in classes:
        j += 1
        condition = len(df1[df1[headers[classID]] == i])
        distribution.append(str(i)+" = "+"{0:.3f}".format(condition/instances))
    return distribution

def kullbackLeibler(n1,n2):
    #Where n1 and n2 are paths to ARFF files
    #It returns the Kullback-Leibler Divergence between the class distributions of n1 and n2
    #Based on: S. Kullback and R. A. Leibler.  On information and sufficiency.
    #The annals of mathematical statistics, 22(1):79–86, 1951.
    data1 = arff.loadarff(n1)
    data2 = arff.loadarff(n2)
    df1 = pd.DataFrame(data1[0])
    df2 = pd.DataFrame(data2[0])
    headers1 = df1.columns
    classID1 = len(df1.columns)-1
    classes1 = df1[headers1[classID1]].unique()
    classes1 = np.sort(classes1)
    instances1=len(df1)
    headers2 = df2.columns
    classID2 = len(df2.columns)-1
    instances2=len(df2)
    prob1 = []
    prob2 = []
    for i in classes1:
        conditon1 = len(df1[df1[headers1[classID1]] == i])
        conditon2 = len(df2[df2[headers2[classID2]] == i])
        prob1.append(conditon1/instances1)
        prob2.append(conditon2/instances2)
    div = 0
    for j in range(len(prob1)):
        div += prob1[j]*math.log(prob1[j]/prob2[j])
    return div

def sim1(n1,n2):
    #Where n1 and n2 are paths to ARFF files
    #It returns value of the metric sim1 
    #Based on: S. Zhang, C. Zhang, and X. Wu. Knowledge Discovery in Multiple Databases.
    #Advanced Information and Knowledge Processing. Springer-Verlag London, Limited, London, UK, 2004
    data1 = arff.loadarff(n1)
    data2 = arff.loadarff(n2)
    df1 = pd.DataFrame(data1[0])
    df2 = pd.DataFrame(data2[0])
    headers1 = df1.columns
    values1 = [] 
    for i in range(len(df1.columns)):
        aux = df1[headers1[i]].unique()
        for j in aux:
            newvalue = str(headers1[i])+"="+str(j)
            if(not(newvalue in values1)):
                values1.append(newvalue)
    headers2 = df2.columns
    values2 = [] 
    for i in range(len(df2.columns)):
        aux = df2[headers2[i]].unique()
        for j in aux:
            newvalue = str(headers2[i])+"="+str(j)
            if(not(newvalue in values2)):
                values2.append(newvalue)
    values3 = [] 
    for i in values1:
        if (i in values2):
            values3.append(i)
    values4 = [] #Union
    values4.extend(values2)
    for i in values1:
        if (not(i in values4)):
            values4.append(i)
    return(len(values3)/len(values4))

def red(n1):
    #Where n1 is the path to a ARFF file
    #It returns the value of the metric Red
    #Based on: M. Møller. Supervised learning on large redundant training sets.
    #International Journal of Neural Systems, 4(1):15–25, 1993.
    data1 = arff.loadarff(n1)
    df1 = pd.DataFrame(data1[0])
    headers1 = df1.columns
    nAttributes = len(headers1)-1 
    classes1 = df1[headers1[-1]].unique()
    nClasses = len(classes1)
    CPE = 0
    denominator = 0
    for i in range(nClasses):
        summation = 0
        classValue = df1[headers1[-1]]==classes1[i]
        for j in range(nAttributes):
            attrib = headers1[j]
            attribValues = df1[headers1[j]].unique()
            nAttribValues = len(attribValues)
            denominator += math.log2(nAttribValues)
            for k in range(len(attribValues)):
                attribValue = df1[attrib]==attribValues[k]
                try:
                    prob = len(df1[attribValue & classValue])/len(df1[classValue])
                    summation += prob*math.log2(prob)
                except ValueError:
                    summation += 0
        CPE += (len(df1[classValue])/len(df1))*summation
    CPE *= -1
    result = (1-(CPE/denominator))
    return result
