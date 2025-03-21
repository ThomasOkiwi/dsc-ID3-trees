# Decision Trees - Introduction
  
## Introduction
In this section, we're going to introduce another kind of model for predicting values that can be used for both continuous and categorical predictions - decision trees. Decision trees are used to classify (or estimate continuous values) by partitioning the sample space as efficiently as possible into sets with similar data points until you get to (or close to) a homogenous set and can reasonably predict the value for new data points.
Despite the fact that they've been around for decades, they are still (in conjunction with ensemble methods that we'll learn about in the next section) one of the most powerful modeling tools available in the field of machine learning. They are also highly interpretable when compared to more complex models (they're simple to explain and it's easy to understand how they make their decisions).
## Entropy and Information Gain
Due to the nature of decision trees, you can get very different predictions depending on what questions you ask and in what order. The question then is how to come up with the right questions to ask in the right order. In this section, we also introduce the idea of entropy and information gain as mechanisms for selecting the most promising questions to ask in a decision tree.
## ID3 Classification Trees
We also talk about Ross Quinlan's ID3 (Iterative Dichotomiser 3) algorithm for generating a decision tree from a dataset.
## Building Trees using Scikit-learn
Next up, we look at how to build a decision tree using the built-in functions available in scikit-learn, and how to test the accuracy of the predictions using a simple accuracy measure, AUC, and a confusion matrix. We also show how to use the graph_viz library to generate a visualization of the resulting decision tree.
## Hyperparameter Tuning and Pruning
We then look at some of the hyperparameters available when optimizing a decision tree. For example, if you're not careful, generated decision trees can lead to overfitting of data (wherein a model is a perfect match for training data, but horrible for test data). There are a number of hyperparameters you can use when generating a tree to minimize overfitting such as maximum depth or minimum leaf sample size. We look at these various "pruning" strategies to avoid overfitting of the data and to create a better model.
## Regression with CART Trees
In addition to building decision tree classifiers, you will also build decision trees for regression problems.

# Introduction to Decision Trees
  
## Introduction
In this lesson, we'll take a look at decision tree classifiers. These are rule-based classifiers and belong to the first generation of modern AI. Despite the fact that this algorithm has been used in practice for decades, its simplicity and effectiveness for routine classification tasks is still on par with more sophisticated approaches. They are quite common in the business world because they have decent effectiveness without sacrificing explainability. Let's get started!
## Objectives
You will be able to:
•	Describe a decision tree algorithm in terms of graph architecture
•	Describe how decision trees are used to create partitions in a sample space
•	Describe the training and prediction process of a decision tree
## From graphs to decision trees
We have seen basic classification algorithms (a.k.a classifiers), including Naive Bayes and logistic regression, in earlier sections. A decision tree is a different type of classifier that performs a recursive partition of the sample space. In this lesson, you will get a conceptual understanding of how this is achieved.
A decision tree comprises of decisions that originate from a chosen point in sample space. If you are familiar with Graph theory, a tree is a directed acyclic graph with a root called "root node" that has no incoming edges. All other nodes have one (and only one) incoming edge. Nodes having outgoing edges are known as internal nodes. All other nodes are called leaves. Nodes with an incoming edge, but no outgoing edges, are called terminal nodes.
## Directed Acyclic Graphs
In computer science and mathematics, a directed graph is a collection of nodes and edges such that edges can be traversed only in a specified direction (eg, from node A to node B, but not from node B to node A). An acyclic graph is a graph such that it is impossible for a node to be visited twice along any path from one node to another. So, a directed acyclic graph (or, a DAG) is a directed graph with no cycles. A DAG has a topological ordering, or, a sequence of the nodes such that every edge is directed from earlier to later in the sequence.
## Partitioning the sample space
So, a decision tree is effectively a DAG, such as the one seen below where each internal node partitions the sample space into two (or more) sub-spaces. These nodes are partitioned according to some discrete function that takes the attributes of the sample space as input.In the simplest and most frequent case, each internal node considers a single attribute so that space is partitioned according to the attribute’s value. In the case of numeric attributes, the condition refers to a range.
![image](https://github.com/user-attachments/assets/40a3cc27-da4e-4c66-b1df-52f9b1434283)

 
This is the basic idea behind decision trees: every internal node checks for a condition and performs a decision, and every terminal node (AKA leaf node) represents a discrete class. Decision tree induction is closely related to rule induction. In essence, a decision tree is a just series of IF-ELSE statements (rules). Each path from the root of a decision tree to one of its leaves can be transformed into a rule simply by combining the decisions along the path to form the antecedent, and taking the leaf’s class prediction as the consequence (IF-ELSE statements follow the form: IF antecedent THEN consequence ).
## Definition
A decision tree is a DAG type of classifier where each internal node represents a choice between a number of alternatives and each leaf node represents a classification. An unknown (or test) instance is routed down the tree according to the values of the attributes in the successive nodes. When the instance reaches a leaf, it is classified according to the label assigned to the corresponded leaf.
![image](https://github.com/user-attachments/assets/30b41db1-c3f2-4961-83a9-9ec9957fb989)

 A real dataset would usually have a lot more features than the example above and will create much bigger trees, but the idea will remain exactly the same. The idea of feature importance is crucial to decision trees, since selecting the correct feature to make a split on will affect the complexity and efficacy of the classification process. Regression trees are represented in the same manner, but instead they predict continuous values like the price of a house.
## Training process
The process of training a decision tree and predicting the target features of a dataset is as follows:
1.	Present a dataset of training examples containing features/predictors and a target (similar to classifiers we have seen earlier).
2.	Train the tree model by making splits for the target using the values of predictors. Which features to use as predictors gets selected following the idea of feature selection and uses measures like "information gain" and "Gini Index". We shall cover these shortly.
3.	The tree is grown until some stopping criteria is achieved. This could be a set depth of the tree or any other similar measure.
4.	Show a new set of features to the tree, with an unknown class and let the example propagate through a trained tree. The resulting leaf node represents the class prediction for this example datum.
 
## Splitting criteria
The training process of a decision tree can be generalized as "recursive binary splitting".
In this procedure, all the features are considered and different split points are tried and tested using some cost function. The split with the lowest cost is selected.
There are a couple of algorithms used to build a decision tree:
•	CART (Classification and Regression Trees) uses the Gini Index as a metric
•	ID3 (Iterative Dichotomiser 3) uses the entropy function and information gain as metrics
## Greedy search
We need to determine the attribute that best classifies the training data, and use this attribute at the root of the tree. At each node, we repeat this process creating further splits, until a leaf node is achieved, i.e., all data gets classified.
This means we are performing a top-down, greedy search through the space of possible decision trees.
In order to identify the best attribute for ID3 classification trees, we use the "information gain" criteria. Information gain (IG) measures how much "information" a feature gives us about the class. Decision trees always try to maximize information gain. So, the attribute with the highest information gain will be split on first.
Let's move on to the next lesson where we shall look into these criteria with simple examples.
## Additional resources
•	R2D3Links to an external site.: This is highly recommended for getting a visual introduction to decision trees. Excellent animations explaining the training and prediction stages shown above.
•	Dataversity: Decision Trees IntroLinks to an external site.: A quick and visual introduction to DTs.
•	Directed Acyclic GraphsLinks to an external site.: This would help relate early understanding of graph computation to decision tree architectures.
Summary
In this lesson, we saw an introduction to decision trees as simple yet effective classifiers. We looked at how decision trees partition the sample space based on learning rules from a given dataset. We also looked at how feature selection for splitting the tree is of such high importance. Next, we shall look at information gain criteria used for feature selection.

# Entropy and Information Gain
  
## Introduction
Information gain is calculated using a statistical measure called Entropy. Entropy is a widely used concept in the fields of Physics, Mathematics, Computer Science (information theory), and more. You may have come across the idea of entropy in thermodynamics, societal dynamics, and a number of other domains. In electronics and computer science, the idea of entropy is usually derived from Shannon's description of entropy to measure the information gain against some cost incurred in the process. In this lesson, we shall look at how this works with the simple example we introduced in the previous lesson.
## Objectives
You will be able to:
•	Explain the process for selecting the best attribute for a split
•	Calculate entropy and information gain by hand for a simple dataset
•	Compare and contrast entropy and information gain
## Shannon's Entropy
#### Entropy is a measure of disorder or uncertainty.
The measure is named after Claude Shannon, who is known as the "father of information theory". Information theory provides measures of uncertainty associated with random variables. These measures help calculate the average information content one is missing when one does not know the value of the random variable. This uncertainty is measured in bits, i.e., the amount of information (in bits) contained per average instance in a stream of instances.
Conceptually, information can be thought of as being stored or transmitted as variables that can take on different values. A variable can be thought of as a unit of storage that can take on, at different times, one of several different specified values, following some process for taking on those values. Informally, we get information from a variable by looking at its value, just as we get information from an email by reading its contents. In the case of the variable, the information is about the process behind the variable.
The entropy of a variable is the "amount of information" contained in the variable.
This amount is not only determined by the number of different values the variable can take, just as the information in an email is not quantified just by the number of words in the email or the different possible words in the language of the email. Informally, the amount of information in an email is proportional to the amount of “surprise” its reading causes.
For example, if an email is simply a repeat of an earlier email, then it is not informative at all. On the other hand, if, for example, the email reveals the outcome of an election, then it is highly informative. Similarly, the information in a variable is tied to the amount of surprise the value of that variable causes when revealed.
Shannon’s entropy quantifies the amount of information in a variable, thus providing the foundation for a theory around the notion of information.
In terms of data, we can informally describe entropy as an indicator of how messy your data is. A high degree of entropy always reflects "messed-up" data with low/no information content. The uncertainty about the content of the data, before viewing the data remains the same (or almost the same) as that before the data was available.
In a nutshell, higher entropy means less predictive power when it comes to doing data science with that data.
## Entropy and decision trees
Decision trees aim to tidy the data by separating the samples and re-grouping them in the classes they belong to.
Because decision trees use a supervised learning approach, we know the target variable of our data. So, we maximize the purity of the classes as much as possible while making splits, aiming to have clarity in the leaf nodes. Remember, it may not be possible to remove the uncertainty totally, i.e., to fully clean up the data. Have a look at the image below:
![image](https://github.com/user-attachments/assets/8e5fb282-8247-4f22-a15e-f941fab54124)

 We can see that the split has not FULLY classified the data above, but the resulting data is tidier than it was before the split. By using a series of such splits that focus on different features, we try to clean up the data as much as possible in the leaf nodes. At each step, we want to decrease the entropy, so entropy is computed before and after the split. If it decreases, the split is retained and we can proceed to the next step, otherwise, we must try to split with another feature or stop this branch (or quit, in which case we claim that the current tree is the best solution).
## Calculating entropy
Let's pretend we have a sample, S. This sample contains N total items falling into two different categories, True and False. Of the N total items we have, n observations have a target value equal to True, and m observations have a target value equal to False. Note that if we know N and n, we can easily calculate m to be m=N−n.
Let's assume our boss brings us the dataset S, and asks us to group each observation in N according to whether their target value is True or False. They also want to know the ratio of Trues to Falses in our dataset. We can calculate this as follows:
p=n/N−(class1)
q=m/N=1−p−(class2)
If we know these ratios, we can calculate the entropy of the dataset S. This will provide us with an easy way to see how organized or disorganized our dataset is. For instance, let's assume that our boss believes that the dataset should mostly be full of "True"'s, with some occasional "False"'s slipping through. The more Falses in with the Trues (or Trues in with the Falses!), the more disorganized our dataset is. We can calculate entropy using the following equation:
E=−p.log2(p)−q.log2(q)
Don't worry too much about this equation yet -- we'll dig deeper into what it means in a minute.
The equation above tells us that a dataset is considered tidy if it only contains one class (i.e. no uncertainty or confusion). If the dataset contains a mix of classes for our target variable, the entropy increases. This is easier to understand when we visualize it. Consider the following graph of entropy in a dataset that has two classes for our target variable:
 ![image](https://github.com/user-attachments/assets/fffe3149-302f-4e3a-84d9-b03761a65b9a)

As you can see, when the classes are split equally, p=0.5 and q=1−p=0.5, the entropy value is at its maximum, 1. Conversely, when the proportion of the split is at 0 (all of one target class) or at 1 (all of the other target class), the entropy value is 0! This means that we can easily think of entropy as follows: the more one-sided the proportion of target classes, the less entropy. Think of a sock drawer that may or may not have some underwear mixed in. If the sock drawer contains only socks (or only underwear), then entropy is 0. If you reach in and pull out an article of clothing, you know exactly what you're going to get. However, if 10% of the items in that sock drawer are actually underwear, you are less certain what that random draw will give you. That uncertainty increases as more and more underwear gets mixed into that sock drawer, right up until there is the exact same amount of socks and underwear in the drawer. When the proportion is exactly equal, you have no way of knowing item of clothing a random draw might give you -- maximum entropy, and perfect chaos!
This is where the logic behind decision trees comes in -- what if we could split the contents of our sock drawer into different subsets, which might divide the drawer into more organized subsets? For instance, let's assume that we've built a laundry robot that can separate items of clothing by color. If a majority of our socks are white, and a majority of our underwear is some other color, then we can safely assume that the two subsets will have a better separation between socks and underwear, even if the original chaotic drawer had a 50/50 mix of the two!
## Generalization of entropy
Now that we have a good real-world example to cling to, let's get back to thinking about the mathematical definition of entropy.
Entropy H(S) is a measure of the amount of uncertainty in the dataset S. We can see this is a measurement or characterization of the amount of information contained within the dataset S.
We saw how to calculate entropy for a two-class variable before. However, in the real world we deal with multiclass problems very often, so it would be a good idea to see a general representation of the formula we saw before. The general representation is:
H(S)=−∑(Pi.log2(Pi))
When H(S)=0, this means that the set S is perfectly classified, meaning that there is no disorganization in our data because all of our data in S is the exact same class. If we know how much entropy exists in a subset (and remember, we can subset our data by just splitting it into 2 or more groups according to whatever metric we choose), then we can easily calculate how much information gain each potential split would give us!
## Information gain
Information gain is an impurity/uncertainty based criterion that uses the entropy as the measure of impurity.
There are several different algorithms out there for creating decision trees. Of those, the ID3 algorithm is one of the most popular. Information gain is the key criterion that is used by the ID3 classification tree algorithm to construct a decision tree. The decision tree algorithm will always try to maximize information gain. The entropy of the dataset is calculated using each attribute, and the attribute showing highest information gain is used to create the split at each node. A simple understanding of information gain can be written as:
Information Gain=Entropyparent−Entropychild.[child weighted average]
A weighted average based on the number of samples in each class is multiplied by the child's entropy, since most datasets have class imbalance. Thus the information gain calculation for each attribute is calculated and compared, and the attribute showing the highest information gain will be selected for the split. Below is a more generalized form of the equation:
When we measure information gain, we're really measuring the difference in entropy from before the split (an untidy sock drawer) to after the split (a group of white socks and underwear, and a group of non-white socks and underwear). Information gain allows us to put a number to exactly how much we've reduced our uncertainty after splitting a dataset S on some attribute, A. The equation for information gain is:
IG(A,S)=H(S)−∑p(t)H(t)
Where:
•	H(S) is the entropy of set S
•	t is a subset of the attributes contained in A (we represent all subsets t as T)
•	p(t) is the proportion of the number of elements in t to the number of elements in S
•	H(t) is the entropy of a given subset t
In the ID3 algorithm, we use entropy to calculate information gain, and then pick the attribute with the largest possible information gain to split our data on at each iteration.
##cEntropy and information gain example
So far, we've focused heavily on the math behind entropy and information gain. This usually makes the calculations look scarier than they actually are. To show that calculating entropy/information gain is actually pretty simple, let's take a look at an example problem -- predicting if we want to play tennis or not, based on the weather, temperature, humidity, and windiness of a given day!
Our dataset is as follows:
![image](https://github.com/user-attachments/assets/1fbdd2b4-d89a-42ab-947b-0905b39f3408)

Let's apply the formulas we saw earlier to this problem:
H(S)=∑−p(c)log2p(c)
C=yes,no
Out of 14 instances, 9 are classified as yes, and 5 as no. So:
p(yes)=−(9/14)log2(9/14)=0.28
p(no)=−(5/14)log2(5/14)=0.37
H(S)=p(yes)+p(no)=0.65
The current entropy of our dataset is 0.65. In the next lesson, we'll see how we can improve this by subsetting our dataset into different groups by calculating the entropy/information gain of each possible split, and then picking the one that performs best until we have a fully fleshed-out decision tree!




# ID3 Classification Trees: Perfect Split with Information Gain - Lab

## Introduction

In this lab, we will simulate the example from the previous lesson in Python. You will write functions to calculate entropy and IG which will be used for calculating these uncertainty measures and deciding upon creating a split using information gain while growing an ID3 classification tree. You will also write a general function that can be used for other (larger) problems as well. So let's get on with it.

## Objectives

In this lab you will: 

- Write functions for calculating entropy and information gain measures  
- Use entropy and information gain to identify the attribute that results in the best split at each node


## Problem

You will use the same problem about deciding whether to go and play tennis on a given day, given the weather conditions. Here is the data from the previous lesson:

|  outlook | temp | humidity | windy | play |
|:--------:|:----:|:--------:|:-----:|:----:|
| overcast | cool |   high   |   Y   |  yes |
| overcast | mild |  normal  |   N   |  yes |
|   sunny  | cool |  normal  |   N   |  yes |
| overcast |  hot |   high   |   Y   |  no  |
|   sunny  |  hot |  normal  |   Y   |  yes |
|   rain   | mild |   high   |   N   |  no  |
|   rain   | cool |  normal  |   N   |  no  |
|   sunny  | mild |   high   |   N   |  yes |
|   sunny  | cool |  normal  |   Y   |  yes |
|   sunny  | mild |  normal  |   Y   |  yes |
| overcast | cool |   high   |   N   |  yes |
|   rain   | cool |   high   |   Y   |  no  |
|   sunny  |  hot |  normal  |   Y   |  no  |
|   sunny  | mild |   high   |   N   |  yes |


## Write a function `entropy(pi)` to calculate total entropy in a given discrete probability distribution `pi`

- The function should take in a probability distribution `pi` as a list of class distributions. This should be a list of two integers, representing how many items are in each class. For example: `[4, 4]` indicates that there are four items in each class, `[10, 0]` indicates that there are 10 items in one class and 0 in the other. 
- Calculate and return entropy according to the formula: $$Entropy(p) = -\sum (P_i . log_2(P_i))$$


```python
from math import log


def entropy(pi):
    """
    return the Entropy of a probability distribution:
    entropy(p) = - SUM (Pi * log(Pi) )
    """

    pass


# Test the function

print(entropy([1, 1]))  # Maximum Entropy e.g. a coin toss
print(
    entropy([0, 6])
)  # No entropy, ignore the -ve with zero , it's there due to log function
print(entropy([2, 10]))  # A random mix of classes

# 1.0
# -0.0
# 0.6500224216483541
```

## Write a function `IG(D,a)` to calculate the information gain 

- As input, the function should take in `D` as a class distribution array for target class, and `a` the class distribution of the attribute to be tested
- Using the `entropy()` function from above, calculate the information gain as:

$$gain(D,A) = Entropy(D) - \sum(\frac{|D_i|}{|D|}.Entropy(D_i))$$

where $D_{i}$ represents distribution of each class in `a`.



```python
def IG(D, a):
    """
    return the information gain:
    gain(D, A) = entropy(D)− SUM( |Di| / |D| * entropy(Di) )
    """

    pass


# Test the function
# Set of example of the dataset - distribution of classes
test_dist = [6, 6]  # Yes, No
# Attribute, number of members (feature)
test_attr = [
    [4, 0],
    [2, 4],
    [0, 2],
]  # class1, class2, class3 of attr1 according to YES/NO classes in test_dist

print(IG(test_dist, test_attr))

# 0.5408520829727552
```

## First iteration - Decide the best split for the root node

- Create the class distribution `play` as a list showing frequencies of both classes from the dataset
- Similarly, create variables for four categorical feature attributes showing the class distribution for each class with respect to the target classes (yes and no)
- Pass the play distribution with each attribute to calculate the information gain


```python
# Your code here


# Information Gain:

print("Information Gain:\n")
print("Outlook:", IG(play, outlook))
print("Temperature:", IG(play, temperature))
print("Humidity:", IG(play, humidity))
print("Wind:,", IG(play, wind))

# Outlook: 0.41265581953400066
# Temperature: 0.09212146003297261
# Humidity: 0.0161116063701896
# Wind:, 0.0161116063701896
```

We see here that the outlook attribute gives us the highest value for information gain, hence we choose this for creating a split at the root node. So far, we've built the following decision tree:
<img src='https://curriculum-content.s3.amazonaws.com/data-science/images/outlook.png'  width ="650"  >


## Second iteration

Since the first iteration determines what split we should make for the root node of our tree, it's pretty simple. Now, we move down to the second level and start finding the optimal split for each of the nodes on this level. The first branch (edge) of three above that leads to the "Sunny" outcome. Of the temperature, humidity and wind attributes, find which one provides the highest information gain.

Follow the same steps as above. Remember, we have 6 positive examples and 1 negative example in the "sunny" branch.


```python
# Your code here


# Information Gain:
print("Information Gain:\n")

print("Temperature:", IG(play, temperature))
print("Humidity:", IG(play, humidity))
print("Wind:,", IG(play, wind))

# Temperature: 0.3059584928680418
# Humidity: 0.0760098536627829
# Wind: 0.12808527889139443
```

We see that temperature gives us the highest information gain, so we'll use it to split our tree as shown below:

<img src='https://curriculum-content.s3.amazonaws.com/data-science/images/temp.png'  width ="650"  >


Let's continue. 

## Third iteration

We'll now calculate splits for the 'temperature' node we just created for days where the weather is sunny. Temperature has three possible values: [Hot, Mild, Cool]. This means that for each of the possible temperatures, we'll need to calculate if splitting on windy or humidity gives us the greatest possible information gain.

Why are we doing this next instead of the rest of the splits on level 2? Because a decision tree is a greedy algorithm, meaning that the next choice is always the one that will give it the greatest information gain. In this case, evaluating the temperature on sunny days gives us the most information gain, so that's where we'll go next.

## All other iterations

What happens once we get down to a 'pure' split? Obviously, we stop splitting. Once that happens, we go back to the highest remaining uncalculated node and calculate the best possible split for that one. We then continue on with that branch, until we have exhausted all possible splits or we run into a split that gives us 'pure' leaves where all 'play=Yes' is on one side of the split, and all 'play=No' is on the other.

## Summary 

This lab should have helped you familiarize yourself with how decision trees work 'under the hood', and demystified how the algorithm actually 'learns' from data by: 

- Calculating entropy and information gain
- Figuring out the next split you should calculate ('greedy' approach) 
