# Import of all needed packages
from contextlib import redirect_stdout
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from nltk.classify import NaiveBayesClassifier as nb
from nltk.metrics import ConfusionMatrix
from nltk.metrics.scores import accuracy as accu
import torch
from torch import nn


def load_data():
    '''
    Load the dataset.
    '''
    print('Loading data...')
    dataset = pd.read_csv('emails.csv/emails.csv')
    return dataset

def preprocessing(dataset):
    '''
    Preprocess the dataset.
    '''

    print('\n\nPreprocessing data...')
    dataset.head()

    # 
    plt.figure(figsize=(10,5))
    spams = dataset['Prediction'].value_counts()
    print(spams)
    plt.hist(dataset['Prediction'])

    plt.title('Spam vs Not_Spam')
    plt.xlabel('Type') 
    plt.ylabel('Count')
    plt.savefig('Spam_vs_not_Spam.png')

    # Preprocessing
    data = []
    for index, row in dataset.iterrows():
        feaututre_dict = {cols: row[cols] for cols in dataset.columns if cols != 'Prediction'}
        labels = row['Prediction']
        data.append((feaututre_dict, labels))
    

    # Splitting the dataset
    train_dataset, test_dataset= train_test_split(data, test_size=0.2, random_state=8)

    print('Dataset processed.')
    print('\nTrain dataset:', len(train_dataset))
    test_labels = [labels for row, labels in test_dataset]
    train_labels = [labels for row, labels in train_dataset]
    print('Train labels:','\nNot spam: ', train_labels.count(0),'\nSpam: ', train_labels.count(1))
    print('\nTest dataset:', len(test_dataset))
    print('Test labels:','\nNot spam: ', test_labels.count(0),'\nSpam: ', test_labels.count(1))
    return train_dataset, test_dataset

def NaiveBayesClassifier(train_dataset, test_dataset):
    ''' 
    Naive Bayes Classifier using nlkt.
    '''

    print('\n\nNaive Bayes Classifier...')
    print('Training...')
    # Training
    classifier = nb.train(train_dataset)

    print('Testing...')
    ros = []
    for row, labels in test_dataset:
        ros.append(row)
    
    # Testing
    out = classifier.classify_many(ros)

    test_labels = [labels for row, labels in test_dataset]

    print('\nMetrics:')
    accuracy = accu(test_labels, out)
    print(f'Accuracy nltk: {accuracy * 100:.2f}%')

    conf_mat = ConfusionMatrix(test_labels, out)
    print(f'Confusion Matrix:\n{conf_mat}')

    from sklearn.metrics import f1_score
    f1 = f1_score(test_labels, out)
    print(f'F1 Score: {f1 * 100:.2f}%')


def FFNN(train_dataset, test_dataset):
    ''' 
    Classifier using a Feed Forward Neural Network.

    Creation of the Neural Network.

    Training and testting.
    '''

    print('\n\nCreating Feed Forward Neural Network...')
    class FFNN(nn.Module):
        def __init__(self, input_dim, hidden_layers, nodes=1000, output_dim=2):
            self.hidden_layers = hidden_layers

            super(FFNN, self).__init__()

            self.fc1 = nn.Linear(input_dim, nodes)
            self.tanh1 = nn.Tanh()

            for i in range(self.hidden_layers):
                layer_name = f'fc{i+2}'
                self.add_module(layer_name, nn.Linear(nodes, nodes))
                self.add_module(f'tanh{i+2}', nn.Tanh())

            self.output = nn.Linear(nodes, output_dim)


        def forward(self, x):
            for i in range(1, self.hidden_layers+2):
                x = getattr(self, f'fc{i}')(x)
                x = getattr(self, f'tanh{i}')(x)
            x = self.output(x)
            return x

    print('Model:')
    model = FFNN(input_dim=3000, hidden_layers=5, output_dim=2)
    print(model)

    train_labels = [labels for row, labels in train_dataset]
    class_weights = torch.tensor([1 - train_labels.count(0) / len(train_labels),
                                   1 - train_labels.count(1) / len(train_labels)])


    # Training
    objective = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print('\nTraining...')
    for l in range(len(train_dataset)):
        if 'Email No.' in train_dataset[l][0]:
            del train_dataset[l][0]['Email No.']
        optimizer.zero_grad()

        out = model(torch.tensor(list(train_dataset[l][0].values()), dtype=torch.float32).unsqueeze(0))

        loss = objective(out, torch.tensor([train_dataset[l][1]]))

        loss.backward()

        optimizer.step()

    
    # Testing

    correct = 0
    true_posistive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    print('Testing...')
    for i in range(len(test_dataset)):
        if 'Email No.' in test_dataset[i][0]:
            del test_dataset[i][0]['Email No.']

        out = model(torch.tensor(list(test_dataset[i][0].values()), dtype=torch.float32).unsqueeze(0))
        out = torch.nn.functional.softmax(out, dim=1)
        correct += (torch.argmax(out) == test_dataset[i][1])

        pred = torch.argmax(out)

        if pred == 0 and 0 == test_dataset[i][1]: true_posistive +=1
        if pred == 1 and 1 == test_dataset[i][1]: true_negative +=1
        if pred == 0 and 0 != test_dataset[i][1]: false_positive +=1
        if pred == 1 and 1 != test_dataset[i][1]: false_negative +=1

    print('\nMetrics:')

    print(f'Accuracy: {correct/len(test_dataset) * 100:.2f}%')

    precision = true_negative / (true_negative + false_positive)
    recall = true_negative / (true_negative + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    confu_matr = np.array([[true_posistive, false_positive], [false_negative, true_negative]])
    print(f'Confusion Matrix:\n{confu_matr}')
    

    print(f'F1: {f1 * 100:.2f}%')


def main():
    print('Spam Detection')
    
    print('Starting...')
    output = open('Results', 'w')
    with output as f:
        with redirect_stdout(f):
            data = load_data()
            train_dataset, test_dataset = preprocessing(data)
            NaiveBayesClassifier(train_dataset, test_dataset)
            FFNN(train_dataset, test_dataset)
        output.close()
    print('Done.')
    print('You can find the results in \'Results.txt\'')

if __name__ == '__main__':
    main()