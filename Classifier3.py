#Dominic Soares
#CPSC 310
#Project - Classifier3 - Dec Tree

import copy
import fileinput
import csv
import sys
import math
from random import randint
import random
import numpy as np
from collections import Counter
from tabulate import tabulate

def main():
    table = read_csv('StarCraft2ReplayAnalysis-Discretized.txt')
    #t = discretize(table, 8*2)
    #write_to_file2(t)
    
    dec_tree(table)


def dec_tree(table):
    """runs classifier over a data tree finds accuracy
        and plots it using tabulate"""
    matrix_league = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    dataTest, dataRemainder = generate_test_and_remainder(table)
    dataTrain = bootStrap(dataRemainder, 20)
    dataAtt = [1,2,3,4,5,6,7,8,9,10,11]
    datax = []
    dataLOL = []
    treea, y= DecisionTree(dataTrain[0][0],datax,dataAtt,dataLOL,0)
    
    Pd = 0 #Positive count
    TPd = 0 #True positive
    TPd1 = 0

    for y in range(len(dataTest)):
        uselessVar, Pd, TPd, TPd1 = guess_league(treea, dataTest[y],Pd,TPd,TPd1)
        actual = int(float(dataTest[y][0]))
        dataTest[y]
        matrix_league[actual-1][int(float(uselessVar))] += 1

    i = 0
    for row in matrix_league:
        matrix_league[i][9] = sum(matrix_league[i])-(i+1)   #total
        if matrix_league[i][9] != 0:
            matrix_league[i][10] = 100*float(matrix_league[i][i+1])/matrix_league[i][9]
        i += 1

    #Print the results from the auto-data DTree
    print "Decision Tree"
    print "Accuracy: ",
    print round(TPd/float(Pd) * 100,2),"%","",
    #print "Error rate: ",
    #print round(abs(100 - (TPd/float(Pd)) * 100),2),"%"
    print ", +-1 league: ",
    print round(TPd1/float(Pd) * 100,2),"%",""

    print tabulate(matrix_league, headers= ['League', 1, 2, 3, 4, 5, 6, 7, 8, "Total", "Recognition "])


def guess_league(tree, instance, P, TP, TP1):
    """get and count accuracies"""
    P += 1
    guess = treeClassifier1(tree,instance)
    guess = int(float(guess))
    actual = int(float(instance[0]))
    if guess == actual:
        TP += 1
    if (guess == actual or guess == actual+1 or guess == actual-1):
        TP1 += 1
    return guess, P, TP, TP1


def treeClassifier(tree, instance):
    """classifier for decision tree"""
    guess = "yes"
    if tree[0] == 'Leaves':
        maxValue = 0
        for i in range(len(tree[1])):
            if tree[1][i][1] > maxValue:
                maxValue = tree[1][i][1]
                guess = tree[1][i][0]      
    else:
        for j in range(len(tree)-2):
            
            try:
                if instance[tree[1]] == tree[j+2][1]:
                    guess = treeClassifier(tree[j+2][2], instance)
            except TypeError:
                pass
    return guess


def treeClassifier1(tree, instance):
    """classifier for decision tree"""
    guess = 5
    if tree[0] == 'Leaves':
        maxValue = 0
        for i in range(len(tree[1])):
            if tree[1][i][1] > maxValue:
                maxValue = tree[1][i][1]
                guess = tree[1][i][0]      
    else:
        for j in range(len(tree)-2):
            try:
                if instance[tree[1]] == tree[j+2][1]:
                    guess = treeClassifier(tree[j+2][2], instance)
            except TypeError:
                pass
    if guess == "yes":
        guess = 5
    return guess


def calc_enew(instances, att_index, class_index):
    """Calculates the E_new of the instance set and returns it"""
    # get the length of the partition
    D = len(instances)
    # calculate the partition stats for att_index (see below)
    freqs = attribute_frequencies(instances, att_index, class_index)
    # find E_new from freqs (calc weighted avg)
    E_new = 0
    for att_val in freqs:
        D_j = float(freqs[att_val][1])
        probs = [(c/D_j) for (_, c) in freqs[att_val][0].items()]
        for p in range(len(probs)):
            if probs[p] == 0:
                probs[p] = 1.0
        E_D_j = -sum([p*math.log(p,2) for p in probs])
        E_new += (D_j/D)*E_D_j
    return E_new


def partStats(table, cLabel):
    """Gets 1-8 when we have a leaf"""
    classVals = list(set(get_column(table, cLabel)))
    stats = []
    stats.append([table[0][cLabel], 1, len(table)])
    for i in range(len(table)-1):
        a = 0
        for j in range(len(stats)):
            if stats[j][0] == table[i+1][cLabel]:
                stats[j][1] += 1
                a = 1
        if a == 0:
            stats.append([table[i+1][cLabel], 1, len(table)])
    return stats #spits back of a list of lists of freqs for various labels.


def get_column(table, ind): #parse all the data into different lists
    """Gets a column in question."""
    listylist = [] #               0
    i = 0            
    for row in table: #Get nice subdivisions
        listylist.append(table[i][ind])   
        i += 1
    i = 0
    return (listylist)#return all of the lists


def attribute_frequencies(instances, att_index, class_index):
    """returns the frequencis of all attributes in the instance set as a dictionary"""
    att_vals = list(set(get_column(instances, att_index))) 
    class_vals = list(set(get_column(instances, class_index)))
    result = {v: [{c: 0 for c in class_vals}, 0] for v in att_vals}
    for row in instances:
        label = row[class_index]
        att_val = row[att_index]
        result[att_val][0][label] += 1
        result[att_val][1] += 1
    return result


def tdidt(instances, att_indexes, att_domains, class_index):
    """Creates a decision tree recursively"""
    if same_class(instances, class_index):
        return instances
    if len(instances) == 0:
        return instances
    min_ent = pick_attribute(instances, att_indexes, class_index)


def isSame(table, classLabel): #checks for uniformity of class label.
    """Checks if all labels are the same."""
    isIt = 0
    original = table[0][classLabel]
    for i in range(len(table)):
        if table[i][classLabel] != original:
            isIt = 1
    return isIt


def DecisionTree(table,x, listOfAttributes, listOfLeaves, cLabel):
    """decision tree algorithm"""
    #Let's build a tree now, it's essentially a list of lists,
    #either with another tree or leaf at each node. 
    if len(listOfAttributes) > 1 and isSame(table, cLabel) == 1:
        
        eList = []
        for i in range(len(listOfAttributes)):
            eList.append(calc_enew(table, i, cLabel))
        
        #print eList
        #print listOfAttributes
        entroMin = min(eList)
        
        for j in range(len(eList)):
            if eList[j] == entroMin:
                whatToRemove = j
        iiq = listOfAttributes[j]#index in question, save in case we need.
        listOfAttributes.remove(listOfAttributes[j])
        eviq = eList[j] #entropy val in question
        eList.remove(eList[j])
        splitVar = j
        
        splitHelper = attribute_frequencies(table, splitVar, cLabel)
        keys = splitHelper.keys()
        #print keys
    
        i = 0
        tablePart = [[] for i in range(len(splitHelper))]
        listOfLeaves.append('Attribute')
        listOfLeaves.append(splitVar)
        for j in range(len(table)):
            for k in range(len(keys)):
                if table[j][splitVar] == keys[k]:
                    tablePart[k].append(table[j])
                    #divide table into partitions.
        for j2 in range(len(tablePart)): 
            bough = []
            listOfLeaves.append(bough)
            bough.append('Value')
            
            bough.append(tablePart[j2][0][splitVar])
            branch = []
            bough.append(branch)
            
            partStats(table, cLabel)
            DecisionTree(tablePart[j2],x,listOfAttributes,branch,cLabel)

        listOfAttributes.append(iiq)
        eList.append(eviq) 
    else:
        #do something with leaves here.
        listOfLeaves.append('Leaves')
        listOfLeaves.append(partStats(table, cLabel))
        x.append(table[0])

    return listOfLeaves, x


def bootStrap(remainder, N): #Takes the big partition and bootstraps.
    """
    remainder: The table of instances not part of the test set.
    N: The number of total decision trees we want.
    """
    listOfTrainVal = []
    length = len(remainder)
    for i in range(N):
        twoPart = [[],[]]
        for j0 in range(length):
            inty = random.randrange(0,length)
            twoPart[0].append(remainder[inty])
        for j1 in range(length):
            isIn = 0
            for k0 in range(len(twoPart[0])):
                if remainder[j1] == twoPart[0][k0]:
                    isIn = 1
            if isIn == 0:
                twoPart[1].append(remainder[j1])
        listOfTrainVal.append(twoPart)
    return listOfTrainVal


def generate_test_and_remainder(table):
    """Returns the test and remainder sets for the random forest classifier"""
    third_of_data = len(table)/3
    test = random_attribute_subset(table, third_of_data)
    remainder = random_attribute_subset(table, 2*third_of_data)
    return test, remainder


def random_attribute_subset(attributes, F):
    """Returns a random subset of size F from the input table"""
    # shuffle and pick first F
    shuffled = attributes[:]  # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]


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


def get_column_as_floats(table, index):
    """Returns all non-null values in table for given index"""
    vals = []
    for row in table:
        if row[index] != "NA":
            vals.append(float(row[index]))
    return vals


def bin_row(values,cutoffs):
    """puts values in bins"""
    values_new = []
    for item in values:
        it = 1   #iterator
        for cutoff in cutoffs:
            if item < cutoff:
                item = it
                values_new.append(item)
                break
            it+=1
    return values_new


def discretize(table, num_bins):
    """take values and bin them to make categorical"""
    temps = []
    temps.append(get_column_as_floats(table,0))
    for i in range(1,12):
        col_values_new = []
        col_values = get_column_as_floats(table,i)
        for item in col_values:
            col_values_new.append(int(float(item)/5)+1)
        temps.append(col_values_new)

    new_table = []
    for j in range(len(table)-2):
        temp_row = []
        for k in range(0,12):
            temp_row.append(temps[k][j])
        new_table.append(temp_row)
    return new_table
        

def equal_width_cutoffs(table, att_index, num_of_bins):
    """get cutoff values"""
    # get the values from the table
    values = get_column_as_floats(table, att_index)
    # find the min and max vals
    min_val = int(float(min(values)))
    max_val = int(float(max(values)))
    # determine the approximate width
    width = int(max_val - min_val) / num_of_bins
    # create and return the cutoff points
    return list(range(min_val + width, max_val + 1, width))


def write_to_file2(table):
    """write the new table to file"""
    tfile = open('StarCraft2ReplayAnalysis-Discretized1.txt', 'w')
    for row in table:
        print>>tfile, row
    tfile.close()

    for line in fileinput.FileInput("StarCraft2ReplayAnalysis-Discretized1.txt",inplace=1):
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace("'","")
        print line,


def write_to_file(table):
    """write the new table to file"""
    tfile = open('StarCraft2ReplayAnalysis-Normalized.txt', 'w')
    for row in table:
        print>>tfile, row
    tfile.close()

    for line in fileinput.FileInput("StarCraft2ReplayAnalysis-Normalized.txt",inplace=1):
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace("'","")
        print line,


def normalize_table(table):
    """normalize attributes 1-11"""
    temps = []
    temps.append(get_column_as_floats(table,0))
    for i in range(1,12):
        col_values = normalize(get_column_as_floats(table,i))
        temps.append(col_values)
    new_table = []
    for j in range(len(table)):
        temp_row = []
        for k in range(0,12):
            temp_row.append(temps[k][j])
        new_table.append(temp_row)
    return new_table


def normalize(xs):
    """normalize between 0-105"""
    minval = min(xs)
    maxval = max(xs)
    minmax = (maxval - minval) * 1.0
    return [((x - minval) / minmax)*105 for x in xs]


if __name__ == '__main__':
    main()

