# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:45:29 2020

@author: HiteshNayak
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint

%matplotlib inline
sns.set_style("darkgrid")

df = pd.read_csv("D:\dell laptop\deconvolut\Machine learning and statistics\decision tree\decision tree from scratch\Iris.csv")
df = df.drop("Id", axis=1)
df = df.rename(columns={"species": "label"})

df.head()

#train test splitting by defining the number of rows we have in test
def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df

#test the function
random.seed(0)
train_df, test_df = train_test_split(df, test_size=20)

#numpy 2d array format , storing the value in object called data
data = train_df.values
data[:5]

#check purity of data as a true false condition 
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

#check    
check_purity(train_df[train_df.PetalWidthCm < 0.8].values)
   
#classify the data
#this is to classify the data interms of figuring out the class with maximum cases satisfying a given condition
#so that we get the split of left side.
def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

#test the function
classify_data(train_df[train_df.PetalWidthCm >1.2].values)

#define potential split conditions
#this will return a dictionary whose keys will be indices of the columns[1,2,3] and values would be list of splits
#potential splits = {2:[2.7,2.7]}
def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):        # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                
                potential_splits[column_index].append(potential_split)
    
    return potential_splits

#test the potential split
potential_splits=get_potential_splits(train_df.values)
train_df.columns
sns.lmplot(data=train_df,x="PetalWidthCm",y="PetalLengthCm",hue="Species",fit_reg=False,size=6,aspect=1.5)
plt.vlines(x=potential_splits[3],ymin=1,ymax=7)

#split the data
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    return data_below, data_above

#check the function
split_column=3
split_value=0.8
data_below,data_above=split_data(data, split_column, split_value)
plotting_df=pd.DataFrame(data,columns=df.columns)
sns.lmplot(data=plotting_df,x="PetalWidthCm",y="PetalLengthCm",hue="Species",fit_reg=False,size=6,aspect=1.5)
plt.vlines(x=split_value,ymin=1,ymax=7)


#calculate entropy
def calculate_entropy(data):
    label_column = data[:, -1]
    #we only need the count of each class and dont need the class values returned by np.unique
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

calculate_entropy(data_below)
calculate_entropy(data_above)

 
#calculate the entropy for below the split and above the split
def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    return overall_entropy

#determine the best split
def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

potential_splits=get_potential_splits(data)
determine_best_split(data, potential_splits)

#decision tree algorithms
sub_tree = {"question": ["yes_answer", "no_answer"]}

example_tree = {"petal_width <= 0.8": ["Iris-setosa", 
                                      {"petal_width <= 1.65": [{"petal_length <= 4.9": ["Iris-versicolor", 
                                                                                        "Iris-virginica"]}, 
                                                                "Iris-virginica"]}]}

#the main decision tree
#df is numpy array so dataframe to numpy conversion we are using this counter
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base cases).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

tree = decision_tree_algorithm(train_df, max_depth=3)
pprint(tree)

#classification
sub_tree
example = test_df.iloc[0]
example 

def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)
    
classify_example(example, tree)

def calculate_accuracy(df, tree):
    #axis=1 so that it works on rows , args is used to pass the second argument (tree) of classify_example we use a , because it becomes tuple and not integer
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["Species"]
    accuracy = df["classification_correct"].mean()
    return accuracy

accuracy = calculate_accuracy(test_df, tree)
accuracy
