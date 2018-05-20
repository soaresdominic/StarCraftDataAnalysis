#Dominic Soares
#CPSC 310
#Project - Classifier2 - Ensemble Regression

import copy
import fileinput
import csv
import sys
from math import sqrt
from random import randint
import numpy as np
from collections import Counter
from tabulate import tabulate

def main():
    table = read_csv('StarCraft2ReplayAnalysis-Processed-SelectAttr.txt')
    get_accuracy_regression(table)


def regression_visual(table):
    """takes 5 random instances in the table and predicts league based on k-nn classifier"""
    league = 0
    actual_league = 0
    for i in range(0,5):
        it = randint(0,count_instances(table)-1)
        instance = table[it]
        league_pred = k_NN_classifier(table,instance)
        league_actual = int(instance[0])
        print "instance:", instance
        print "league:", league_pred, "actual league:", league_actual


def get_accuracy_regression(table):
    """runs regressions over specific number of instances, finds accuracy
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
    #change upper bound for number of instances to test
    for i in range(0,1500):
        it = randint(0,count_instances(table)-1)
        instance = table[it]
        apm = float(instance[1])
        hotkey = float(instance[3])
        latency = float(instance[9])
        #run regressions, returns what each predicted
        guess = pred_league(table,instance,apm,hotkey,latency)
        if(len(guess) > 2):     #if all different guess, guess middle
            guess = guess[1][0]
        else:   #majority voting
            guess = guess[0][0]

        actual = int(instance[0])
        matrix_league[actual-1][guess] += 1
        
        if(guess == actual):
            correct+=1
        else:
            wrong+=1
        #+- 1 league
        if(guess == actual or guess == actual+1 or guess == actual-1):
            correct1+=1
        else:
            wrong1+=1
        
    accuracy = (1.0*correct)/(1.0*(correct+wrong))
    accuracy = round(accuracy*100,2)
    accuracy1 = (1.0*correct1)/(1.0*(correct1+wrong1))
    accuracy1 = round(accuracy1*100,2)
    print "Regression Ensemble"
    print "Accuracy:", accuracy,"%, ", "+-1 league:", accuracy1,"%"
    print "Error Rate:", 100-accuracy,"%, ", "+-1 league:", 100-accuracy1,"%"

    i = 0
    for row in matrix_league:
        matrix_league[i][9] = sum(matrix_league[i])-(i+1)   #total
        if matrix_league[i][9] != 0:
            matrix_league[i][10] = 100*float(matrix_league[i][i+1])/matrix_league[i][9]
        i += 1

    print tabulate(matrix_league, headers= ['League', 1, 2, 3, 4, 5, 6, 7, 8, "Total", "Recognition "])


def pred_league(table,instance,apm,hotkey,latency):
    """runs regression for each attribute for each equation"""
    apm_pred = regression_apm(table,apm)
    latency_pred = regression_latency(table,latency) 
    hotkey_pred = regression_hotkey(table,hotkey)
    leagues = [apm_pred,latency_pred,hotkey_pred]
    data = Counter(leagues)
    return data.most_common(3)      #return predictions in counter format


#quadratic regression analysis
def regression_hotkey(table, hotkeys):
    """does regression for hotkey attribute based off equation from MATLAB"""
    differences = []
    for x in range(1,9):
        y = (1.6E-05*(x**2))-(4E-05*x)+.00023
        differences.append(abs(y-float(hotkeys)))
    iterator = 1
    for diff in differences:
        if diff == min(differences):
            return iterator
        iterator += 1


def regression_latency(table, latency):
    """does regression for latency attribute based off equation from MATLAB"""
    differences = []
    for x in range(1,9):
        y = (.39*(x**2))-(12*x)+110
        differences.append(abs(y-float(latency)))
    iterator = 1
    for diff in differences:
        if diff == min(differences):
            return iterator
        iterator += 1


def regression_apm(table, apm):
    """does regression for apm attribute based off equation from MATLAB"""
    differences = []
    for x in range(1,9):
        y = (3.7*(x**2))-(6.3*x)+69
        differences.append(abs(y-float(apm)))
    iterator = 1
    for diff in differences:
        if diff == min(differences):
            return iterator
        iterator += 1


def plot_means(table,index):
    """plot the mean of the attribute for each league.
        points are entered in MATLAB for nearest quadratic equations"""
    means = {}
    for i in range(1,9):
        vals = []
        for row in table:
            if(int(float(row[0])) == i):
                vals.append(float(row[index]))
        means[i] = sum(vals)/float(len(vals))
    print means
    leagues = means.keys()
    values = means.values()
    
    fig = plt.figure()
    plt.scatter(leagues,values)
    plt.ylabel('Action Latency')
    plt.xlabel('League')
    plt.title('Action Latency vs League')
    plt.grid(True)
    plt.savefig('Action Latency.pdf')
    plt.close(fig)


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


def get_column_as_floats(table, index):
    """Returns all non-null values in table for given index"""
    vals = []
    for row in table:
        if row[index] != "NA":
            vals.append(float(row[index]))
    return vals


if __name__ == '__main__':
    main()

