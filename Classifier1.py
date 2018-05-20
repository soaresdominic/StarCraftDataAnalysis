#Dominic Soares
#CPSC 310
#Project - Classifier1 - KNN

import copy
import fileinput
import csv
import sys
from math import sqrt
from random import randint
import numpy as np
from tabulate import tabulate

def main():
    table = read_csv('StarCraft2ReplayAnalysis-Processed-SelectAttr.txt')
    get_accuracy_knn(table,50)     #2nd param changes number of instances to test
    #predict_league_knn_visual(table)

def get_accuracy_knn(table,num_instances):
    """runs KNN over specific number of instances, finds accuracy
        and plots it using tabulate"""
    matrix_league = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    correct = 0
    wrong = 0
    correct1 = 0
    wrong1 = 0
    for i in range(0,num_instances):
        it = randint(0,count_instances(table)-1)
        instance = table[it]
        pred_league = predict_league_knn(table,instance)
        actual = int(instance[0])

        #fill in matrix with prediction
        matrix_league[actual-1][pred_league] += 1
        
        if(pred_league == actual):
            correct+=1
        else:
            wrong+=1
        if(pred_league == actual or pred_league == actual+1 or pred_league == actual-1):
            correct1+=1
        else:
            wrong1+=1
        
    accuracy = (1.0*correct)/(1.0*(correct+wrong))
    accuracy = round(accuracy*100,2)
    accuracy1 = (1.0*correct1)/(1.0*(correct1+wrong1))
    accuracy1 = round(accuracy1*100,2)
    print "K-NN,", num_instances,"instances"
    print "Accuracy:", accuracy,"%, ", "+-1 league:", accuracy1,"%"
    print "Error Rate:", 100-accuracy,"%, ", "+-1 league:", 100-accuracy1,"%"

    i = 0
    #add total and recognition rates to matrix
    for row in matrix_league:
        matrix_league[i][9] = sum(matrix_league[i])-(i+1)   #total
        if matrix_league[i][9] != 0:
            matrix_league[i][10] = 100*float(matrix_league[i][i+1])/matrix_league[i][9]
        i += 1
    print tabulate(matrix_league, headers= ['League', 1, 2, 3, 4, 5, 6, 7, 8, "Total", "Recognition "])



def predict_league_knn(table,instance):
    """returns predicted league which is average of predictions"""
    k_NN = k_NN_classifier(table,instance)
    k_NN = [float(i) for i in k_NN]     #turn nums into floats for math
    league_avg = sum(k_NN)/len(k_NN)
    league_pred = round(league_avg)
    return int(league_pred)


def predict_league_knn_visual(table):
    """takes 5 random instances and visually shows KNN prediction"""
    knn_class = 0
    actual_class = 0
    mpg_avg = 0
    for i in range(0,5):
        it = randint(0,count_instances(table)-1)
        instance = table[it]

        k_NN = k_NN_classifier(table,instance)
        k_NN = [float(i) for i in k_NN]     #turn nums into floats for math
        league_avg = sum(k_NN)/len(k_NN)
        league_pred = round(league_avg)
        
        league_actual = int(instance[0])
        print "instance:", instance
        print "league:", league_pred, "actual league:", league_actual


def k_NN_classifier(table, instance):
    """finds euclidian distance to classify league based on all other attributes"""
    k_NN_list = []
    table_kNN = copy.deepcopy(table)    #mirror table to modify 

    for i in range(0,10):  #k=5
        minDis = 10000.0
        minDisInstance = []
        
        for row in table_kNN:
            if (row == instance):
                continue
            #euclidian distance
            temp_d = ((float(row[1])-float(instance[1]))**2)+((float(row[2])-float(instance[2]))**2)
            temp_d1 = ((float(row[3])-float(instance[3]))**2)+((float(row[4])-float(instance[4]))**2)
            temp_d2 = ((float(row[5])-float(instance[5]))**2)+((float(row[6])-float(instance[6]))**2)
            temp_d3 = ((float(row[7])-float(instance[7]))**2)+((float(row[8])-float(instance[8]))**2)
            temp_d4 = ((float(row[9])-float(instance[9]))**2)+((float(row[10])-float(instance[10]))**2)
            temp_d5 = ((float(row[11])-float(instance[11]))**2)
            temp = sqrt(temp_d + temp_d1 + temp_d2 + temp_d3 + temp_d4 + temp_d5)
            
            if(temp < minDis):
                minDis = temp
                minDisInstance = row
        k_NN_list.append(minDisInstance[0])
        table_kNN.remove(minDisInstance)
    return k_NN_list


def read_csv(filename):
    """Reads in a csv file and returns a table as a list of lists (rows)"""
    the_file = open(filename, 'r')
    the_reader = csv.reader(the_file, dialect='excel')
    table = []
    subset_limit = 0
    for row in the_reader:
        if subset_limit >= 5000:
            break
        elif len(row) > 0:
            table.append(row)
        subset_limit += 1
    the_file.close()
    return table


def count_instances(table):
    """counts the number of instances in the table"""
    instances = 0
    for row in table:
        instances += 1
    return instances


if __name__ == '__main__':
    main()

