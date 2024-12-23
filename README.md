## Assignment 1. 

The program made in this assignment is a classification of email in _Spam_ and _Not spam_.

The program use the given [dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv) to train the model and then test it.

### Files:

The folder: **emails.csv** contains the dataset used in this assignment.

**packages** contains all the pacages used. It contains more than the necessary packages.

The necessary packages are:
- pandas
- numpy
- sklearn
- nltk
- matplotlib
- torch
- contextlib

**spam_detection.py** is the main file of the program. It contains the code to train and test the model.

**Spam_vs_not_Spam** it is an histogram that shows the number of spam and not spam emails in the dataset.

**results** contains the results of the program. It contains the confusion matrix and the accuracy of models.

### The program:

The program is divided in 3 parts:
1. Data preprocessing
2. Training of the choosen model
3. Testing of the model

There are 2 models used in this program:
- Naive Bayes, trained using nltk
- Neural Network, trained using torch

### Results:

For the given assignment the results are below:

- Naive Bayes:
    - Accuracy: 87.83%
    - F1 Score: 77.74%
    - Confusion Matrix:
    
    [[689 37 ]
    
    [89 220]]



- Neural Network:
    - Accuracy: 72.66%
    - F1: 67.13%
    - Confusion Matrix:

[[463  20]

 [263 289]]

    