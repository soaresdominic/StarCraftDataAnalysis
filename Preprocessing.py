#Dominic Soares
#CPSC 310
#Project - Preprocessing

import copy
import fileinput
import csv
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def main():
    raw_data = read_csv('StarCraft2ReplayAnalysis-Processed.txt')
    raw_data.pop(0)
    create_frequency_diagram(raw_data, 1, "League", "Number of Instances per League", "IPL.pdf")

    create_boxplot(raw_data,1,2,"Age","League","Age.pdf")
    create_boxplot(raw_data,1,3,"HoursPerWeek","League","HoursPerWeek.pdf")
    create_boxplot(raw_data,1,4,"TotalHours","League","TotalHours.pdf")
    create_boxplot(raw_data,1,5,"APM","League","APM.pdf")
    create_boxplot(raw_data,1,6,"SelectByHotkeys","League","SelectByHotkeys.pdf")
    create_boxplot(raw_data,1,7,"AssignToHotkeys","League","AssignToHotkeys.pdf")
    create_boxplot(raw_data,1,8,"UniqueHotkeys","League","UniqueHotkeys.pdf")
    create_boxplot(raw_data,1,9,"MinimapAttacks","League","MinimapAttacks.pdf")
    create_boxplot(raw_data,1,10,"MinimapRightClicks","League","MinimapRightClicks.pdf")
    create_boxplot(raw_data,1,11,"NumberOfPACs","League","NumberOfPACs.pdf")
    create_boxplot(raw_data,1,12,"GapBetweenPACs","League","GapBetweenPACs.pdf")
    create_boxplot(raw_data,1,13,"ActionLatency","League","ActionLatency.pdf")
    create_boxplot(raw_data,1,14,"ActionsInPAC","League","ActionsInPAC.pdf")
    create_boxplot(raw_data,1,15,"TotalMapExplored","League","TotalMapExplored.pdf")
    create_boxplot(raw_data,1,16,"WorkersMade","League","WorkersMade.pdf")
    create_boxplot(raw_data,1,17,"UniqueUnitsMade","League","UniqueUnitsMade.pdf")
    create_boxplot(raw_data,1,18,"ComplexUnitsMade","League","ComplexUnitsMade.pdf")
    create_boxplot(raw_data,1,19,"ComplexAbilityUsed","League","ComplexAbilityUsed.pdf")
    create_boxplot(raw_data,1,20,"MaxTimeStamp","League","MaxTimeStamp.pdf")

    table = del_useless_attributes(raw_data)
    table_nohours = split_table_hoursperweek(table)
    write_to_file(table_nohours)


def create_frequency_diagram(table, index, x_title, title, filename):
    """creates a bar chart with the number of a specific categorical attribute"""
    fig = plt.figure()    
    mpg = count_items(table, index)  #dictionary of vals - attribute:count

    xs = mpg.keys()     #each attribute
    ys = mpg.values()   #count for each attribute

    plt.bar(xs, ys)
    
    plt.xlabel(x_title)
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close(fig)


def count_items(table, index):
    """returns dictionary of attributes and their counts"""
    items = get_column_as_floats(table, index)
    index = {}
    
    for item in items:
        if(index.has_key(item)):
            index[item] = index[item]+1
        else:
            index[item] = 1
    return index


def write_to_file(table):
    """write the new table to file"""
    tfile = open('StarCraft2ReplayAnalysis-Processed-SelectAttr.txt', 'w')
    for row in table:
        print>>tfile, row
    tfile.close()

    #get rid of all chars not numbers or ,
    for line in fileinput.FileInput("StarCraft2ReplayAnalysis-Processed-SelectAttr.txt",inplace=1):
        line = line.replace("[","")
        line = line.replace("]","")
        line = line.replace("'","")
        print line,


def del_useless_attributes(table):
    """delete attributes with data that doesnt vary league to league"""
    data = copy.deepcopy(table)
    for row in data:
        row.pop(0)
        row.pop(1)
        row.pop(2)
        row.pop(13)
        row.pop(13)
        row.pop(13)
        row.pop(13)
        row.pop(13)
    return data    
    

def split_table_hoursperweek(table):
    """delete hours per week attribute"""
    data = copy.deepcopy(table)
    for row in data:
        row.pop(1)
    return data  
    

def create_boxplot(table, x_index, y_index, ylabel, xlabel, filename):
    """creates a boxplot"""
    fig = plt.figure()
    num_instances = count_instances(table)

    league_list = get_column_as_floats(table, x_index)
    leagues = set(league_list)
    leagues = list(leagues)
    leagues_unique = [int(i) for i in leagues]  #x vals
    
    #lists that have the values for each league
    data = [[] for i in range(len(leagues_unique))]
    it = 1
    list_it = 0
    for i in range(0,len(leagues_unique)):
        for row in table:
            if(int(row[x_index]) == it):
                data[list_it].append(float(row[y_index]))
        it += 1
        list_it += 1
    plt.boxplot(data)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(ylabel + " by League")
    data_range = list(range(1,len(leagues_unique)+1))
    plt.xticks(data_range, leagues_unique)
    plt.grid(True)

    plt.savefig(filename)
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

