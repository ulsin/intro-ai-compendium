# Table of Contents

- [Table of Contents](#table-of-contents)
- [Intro](#intro)
  - [General Programming vs AI](#general-programming-vs-ai)
  - [Steps to Design an AI system](#steps-to-design-an-ai-system)
  - [Daily life of an A.I programmer](#daily-life-of-an-ai-programmer)
- [Data](#data)
  - [Data pitfalls (problems which can occur with data)](#data-pitfalls-problems-which-can-occur-with-data)
  - [How to work with data ?](#how-to-work-with-data-)
    - [1. Data labeling / annotation](#1-data-labeling--annotation)
    - [2. Data annonimization](#2-data-annonimization)
    - [3. Synthetic data](#3-synthetic-data)
    - [4. Data preperation](#4-data-preperation)
    - [5. Data wrangling](#5-data-wrangling)
    - [6. Data mining](#6-data-mining)
    - [7. Data warehousing](#7-data-warehousing)
    - [8. Data Engineering](#8-data-engineering)
- [Machine Learning](#machine-learning)
  - [Unsupervised](#unsupervised)
  - [Supervised](#supervised)
    - [Classification](#classification)
      - [Types](#types)
        - [Support vector machines](#support-vector-machines)
        - [Logistic regression (This is not a regression algorithm)](#logistic-regression-this-is-not-a-regression-algorithm)
        - [Naïve Bayes classifier](#naïve-bayes-classifier)
      - [Classification versus Clustering algorithms](#classification-versus-clustering-algorithms)
    - [Regression](#regression)
      - [Types](#types-1)
        - [Linear Regression](#linear-regression)
        - [Polynomial Regression](#polynomial-regression)
  - [Reinforcement learning](#reinforcement-learning)
- [Limitations](#limitations)

# Intro
- AI smart and dumb at once
- AI is when we artificially introduce intelligence in machines
- Solves Problme without human interaction


## General Programming vs AI
| Conventional programming                                                                                             | Artificial Intelligence                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Programmers look at the problem (desired output) and build an algorithm/application to solve this problem.           | Ai programmers show the problem (desired output) to the A.I algorithms and expect the algorithms to find a solution.     |
| A programmer has complete control over their application                                                             | An A.I programmer can never claim to have full control over their A.I applications. (Explainable A.I is hard to achieve) |
| The software must follow a logical series of steps to reach a conclusion (hard coded instructions by the programmer) | Ai applications use the technique of search and pattern matching                                                         |
| Its easy to explain a conventional algorithm                                                                         | Its very hard to explain how an A.I algorithm reached its desired output                                                 |
| The most important element here is the algorithm                                                                     | The most important elements here are data and algorithms                                                                 |

<p align="center">
    <img src="pictures/general-vs-ai.png" style="width: auto;" alt="">
</p>

- 4th, latest and biggest industrial revolutions

- 1955, coined AI term

3 Types of AI
- Narrow (the ones we have today)
  - Dedicated to assist wiht or take over specific tasks

- General AI (Human like in capabilities)
  - Takes knowledge from one domain and transfers to other domain

- Super AI (Currently fictional, ie. The Matrix, Skynet, BLAME!)
  - Machines that are an order of magnitude smarter than humans

- Turring test (1950)
  - A test a computer can pass, to try and pass as a human
  - Google Duplex has passed this

AI has helped with:
* Human trafficking
* Money Laundering
* Terrorism
* Covid19 research
* etc ..

## Steps to Design an AI system
1. Identify the problem
2. Prepare the data
3. Choose the algorithms
4. Train the algorithms with the data
5. Run on a selected platform

## Daily life of an A.I programmer
* 80% time spend on data (cleaning, preparing, labeling, analyzing etc)
* 5% on deployment (cloud/on premise)
* 15% on A.I development

# Data

Two main points
- Data clearning
- Feature Engineering

<p align="center">
    <img src="pictures/data.webp" style="width: auto;" alt="">
</p>

<p align="center">
    <img src="pictures/data-percentage-of-time.png" style="width: auto;" alt="">
</p>

Data is the new most valuable resource
Data is the new oil

## Data pitfalls (problems which can occur with data)
* Assuming the data is clean
  * e.g spelling  mistakes
* Outliers
  * Excluding  outliers
  * Including  outliers
* Ignroing seasonality
  * Easter vacations,  summer holidays,  black Friday etc.
* Context is critical
  * Ignoring  size when reporting  growth
* Poor data Insights
* Not connecting with external data
* Lacking business understanding

## How to work with data ?
### 1. Data labeling / annotation
- Setting labels on certain features of the data
- Can be aided by clustering on unlabeled data to accelerate the process

### 2. Data annonimization
- Faking atributes to similar but non identifying values, to retain the patterns of the data without the identifying information

### 3. Synthetic data
- Generating data to aid with padding a data set

### 4. Data preperation
This is done before the data is used for training the model, to optimise the quality of the model. Better data = Better model  

<p align="center">
    <img src="pictures/what-makes-good-model.png" style="width: auto;" alt="">
</p>

**Data cleansing  + feature engineering**  
* Feature  engineering,  also known as feature creation,  is the process  of constructing  new  features from existing  data  to train a machine learning  model.

**For example**  
* Character recognition
  * Features may include  histograms counting  the number of black pixels  along horizontal and vertical directions,  number of internal  holes,  stroke detection and many others. 
* Speech recognition
  * Features for recognizing  phonemes can include  noise ratios,  length of sounds,  relative power, filter matches and many others. 
* Spam detection
  * Features may include  the presence  or absence of certain email headers,  the email structure,  the language,  the frequency  of specific terms, the grammatical correctness  of the text. 
* Computer vision
  * There are a large number of possible  features, such  as edges and objects. 

**Feature Extraction**  
Extracting sub-freatures from columns(in this context features) of the data set, which can be extracted manually *(Think 3NF hahah)*
- Location
  - Adress, City, State and Zip code
- Time
  - Year, month day hour minute time ranges (numeric)
  - Weekdays or weekend (binary true / false state)
- Numbers
  - Age numbers can be turned into ranges (oridnal or categorical)

**Transformations**  
Create new features out of existing data  

**Aggregations**  
Create features which are based on combining other features in some way (grouping or putting some mathematical funcion on them)  

### 5. Data wrangling
The process of transforming and mapping data from one "raw" data form into another format with the intent of making it more appropriate and valuable for a variety of downstream purposes such as analytics.

### 6. Data mining
The process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems.

<p align="center">
    <img src="pictures/data-mining-process.png" style="width: auto;" alt="">
</p>

### 7. Data warehousing
  * ETL programming

<p align="center">
    <img src="pictures/data-warehousing.png" style="width: auto;" alt="">
</p>

### 8. Data Engineering
  * Infrastructure,  big data, cloud etc..

<p align="center">
    <img src="pictures/data-hierarchy-of-needs.png" style="width: auto;" alt="">
</p>



# Machine Learning

<p align="center">
    <img src="pictures/MLxkcdMap.jpg" style="width: auto;" alt="">
</p>

<p align="center">
    <img src="pictures/ML-trio.png" style="width: auto;" alt="">
</p>

<p align="center">
    <img src="pictures/scikit-chooser.png" style="width: auto;" alt="">
</p>

<p align="center">
    <img src="pictures/MLmap3.png" style="width: auto;" alt="">
</p>

## Unsupervised

<p align="center">
    <img src="pictures/unsupervisedML.png" style="width: auto;" alt="">
</p>

* Clustering
  - Recomender Systems
    * Recommender systems are an important class of machine learning algorithms that offer "relevant" suggestions to users.
    * These systems predict the most likely product that the users are most likely to purchase and are of interest to them
  - Targeted Markeding
  - Customer Segmenation
* Dimensionality Reduction
  - Meaningful Compression
  - Structure Discovery
  - Big Data Visualisation
  - Feature Elicitation

## Supervised
Two Types, Classification and Regression

<p align="center">
    <img src="pictures/supervisedML.png" style="width: auto;" alt="">
</p>

<p align="center">
    <img src="pictures/classification-vs-regression.png" style="width: auto;" alt="">
</p>


| Regression Algorithm                                                                                                        | Classification Algorithm                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| In Regression, the output variable must be of continuous nature or real value.                                              | In Classification, the output variable must be a discrete value.                                                                                                       |
| The task of the regression algorithm is to map the input value (x) with the continuous output variable(y).                  | The task of the classification algorithm is to map the input value(x) with the discrete output variable(y).                                                            |
| Regression Algorithms are used with continuous data.                                                                        | Classification Algorithms are used with discrete data.                                                                                                                 |
| Regression algorithms can be used to solve the regression problems such as Weather Prediction, House price prediction, etc. | Classification Algorithms can be used to solve classification problems such as Identification of spam emails, Speech Recognition, Identification of cancer cells, etc. |
| The regression Algorithm can be further divided into Linear and Non-linear Regression.                                      | The Classification algorithms can be divided into Binary Classifier and Multi-class Classifier.                                                                        |


### Classification

<p align="center">
    <img src="pictures/classification.png" style="width: auto;" alt="">
</p>

* Divides the data in classes / categories
* Use cases:
  * Spam detection (classification: spam or normal)
  * Analysis of the customer data to predict whether they will buy computer accessories (classification: Yes or No)
  * Classifying fruits from features like colour, taste, size, weight (classification: Apple, Orange, Cherry, Banana)
  * Gender classification from hair length (classification: Male or Female)
  * Stock market price prediction (classification: high or low)

**Aplications**
- Image Classification
  - ML for detecing cat pictures
  - Collect Data (Pictures of cats and not cats)
  - Label Data (Clustering could help with labeling) (Label images with cats, and those without cats)
  - Train Model with labeled data
  - After modeled is trained, give it new images to test and see if it gives right answer

<p align="center">
    <img src="pictures/ML-flow.png" style="width: auto;" alt="">
</p>

  - Customer Retention
  - Identity Frad Detection
  - Diagnostics

#### Types
* Decision Trees
* K-Nearest Neighbour
* Random Forest

##### Support vector machines

<p align="center">
    <img src="pictures/support-vector.png" style="width: auto;" alt="">
</p>

* Main use is to classify unseen data
* Supprt vector machines classifies the data by finding a clear seperation between the data points
  * It looks for a hyperplane
* Can be used both in classification and regression

**Hyperplanes**
- 3D vector for 3D data
  - Or just higher dimension plane same dimension of data
  - n dimension vector for n dimensoin data

**Uses of support vector machines**

<p align="center">
    <img src="pictures/support-vector.png" style="width: auto;" alt="">
</p>

* Used to detect cancerous cells
* Used to predict driving routes
* Face detection
* Image classification
* Handwriting detection

**Pros**
* Effective on data sets with multiple features
* Effective in cases where number of features is greater than the data points
* Memory efficient

**Cons**
* They donot provide probability estimates. Those are calculated using an 
expensive five fold cross vsalidation
* Works best on small sample sets because of its high training time

##### Logistic regression (This is not a regression algorithm)
* is used when you have a classification problem - yes/no, pass/fail, win/lose, alive/dead, healthy/sick etc..
* It is the go-to method for binary classification problems 
* The logistic function looks like a big S and will transform any value into the range 0 to 1.
* Sigmoid function:
  * Allows to put a threshold value. e.g 0.5 (use case spam detection)

**Uses**  
* Logistic regression is used to predict the occurance of some event. e.g 
* Predict whether rain will occur or not
* spam detection, Diabetes prediction, cancer detection etc.

**Types of Logistic Regression**
* Binary logistic regression (e.g pass / fail)
* Multiclass logistic regression (e.g cats, dogs, sheep)
* Ordinal (low, medium, high)

<p align="center">
    <img src="pictures/logistic-regression.png" style="width: auto;" alt="">
</p>

**Example for Binary Logistic Regression**
1. Visualize
<p align="center">
    <img src="pictures/binary-logistic-1.png" style="width: auto;" alt="">
    <img src="pictures/binary-logistic-2.png" style="width: auto;" alt="">
</p>

2. Sigmoid Activation
  - Making a sigmoid curve (s-curve) which defines to classification binary for making the binary descicion
3. Descision boundary
  - Setting a threshold value
4. Making predictions
  - Based on a value from the function we can place on either side of the boundary
5. Cost function
  - Measures how wrong hte model is in terms of ability to estimate relationship between two values
  - Gradient Descent
6. Map probabbilities to classes
  - Based on the dividing line, it is either class A or class B, (or more but in this case it is binary)
7. Finish Up
* Train the model
* Evaluate the model
  * Minimzie the cost with repeated iterations
* Measure the accuracy of your outputs
* Meausre the probabbility score


##### Naïve Bayes classifier
* A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
* Naive bayes does not take into account the relationship between features
* e.g a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter.
* Its an easy to build algorithm and a very powerful one.

**For Example**  
* A dog will be considered a dog because it has 4 legs and a tail
* But a Naive Bayes will consider everything a dog which has 4 legs or tail (e.g squirrel) not because its an animal. Being an animal is the relationship not legs / tail.
* A fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter.

**Naive Bayes use in Natural Language Processing**



**Bayres Theorem**  
Bayes's theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event

<p align="center">
    <img src="pictures/bayes-theorem.png" style="width: auto;" alt="">
</p>

<p align="center">
    <img src="pictures/bayes-table.png" style="width: auto;" alt="">
</p>

We can see that  
  * 50% of the fruits are bananas
  * 30% are oranges
  * 20% are other fruits  

We can also say  
  * From 500 bananas 400 (0.8) are Long, 350 (0.7) are Sweet and 450 (0.9) are Yellow
  * Out of 300 oranges 0 are Long, 150 (0.5) are Sweet and 300 (1) are Yellow
  * From the remaining 200 fruits, 100 (0.5) are Long, 150 (0.75) are Sweet and 50 (0.25) are Yellow




#### Classification versus Clustering algorithms
* Clustering groups similar kind of things together (And is unsupervised)
* However classification predicts similar kind of things together based on the historic trained data (and is supervised)

### Regression
* Regression models are used to predict a continuous value
* Regression is used for prediction, forecasting, time series modeling, and determining the causal effect between variables
* Examples
  * Predicting  prices of a house  given a set of features (size, price, location  etc)
  * Predicting  sales revenue  of a company based on previous  sales figures

<p align="center">
    <img src="pictures/regression-plot.png" style="width: auto;" alt="">
</p>


**Aplications**
- Advertising Popularity Prediction
- Weather Forecasting
- Market Forecasting
- Estimating Life expectancy
- Population Growth Prediction

#### Types
Minor types
* Logistic regression
* Support vector regression
* Decision tree regression
* Random forest regression
* Ridge regression
* Lasso regression


Important types

##### Linear Regression
* It’s a linear model
  * an equation that describes a relationship between two quantities that show a constant rate of change.
  * e.g the older I get, the wiser I will be (hopefully)
  * Attending all lectures in Intro to A.I course will result in passing the exam
* There is always an input variable (x) and an output variable (y)
  * Y can be calculated from a linear combination of input variables (x)
* Two Types
  * Simple Linera Regression (single input value)
  * Muliple Linera Regression (multiple input value)

<p align="center">
    <img src="pictures/linear-regression.png" style="width: auto;" alt="">
</p>



##### Polynomial Regression
- Also Linear, but curved line (like polynomial function hahah)
- Can create a better fit on values that seem to be curving

<p align="center">
    <img src="pictures/lnear-vs-poly-regression.png" style="width: auto;" alt="">
</p>


## Reinforcement learning

<p align="center">
    <img src="pictures/reinforcementChart.png" style="width: auto;" alt="">
</p>

**Aplications**
* Game AI (Guessing the sort where an AI plays a game)
* Skill Acuisition
* Learning Tasks
* Robot Navigation
* Real-time desciscions

* Reinforcement Learning is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences.
* Whenever the model predicts or produces a result, it is penalized if the prediction is wrong or rewarded if the prediction is correct.

**Challenges with Reinforcement learning**
* The main challenge in reinforcement learning lays in preparing the simulation environment, which is highly dependant on the task to be performed. 
* Scaling and tweaking the neural network controlling the agent is another challenge. There is no way to communicate with the network other than through the system of rewards and penalties.

# Limitations