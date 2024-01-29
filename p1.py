# from random import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

drugs_csv = pd.read_csv('drug.csv')

# missing values and its number
drugs_csv_missing = drugs_csv.isnull().sum()
print("Missing Values in Drugs file are:\n", drugs_csv_missing)
print("_" * 100)

# handling Missing values by replacing categorical columns with mode and numerical columns with mean
drugs_csv['BP'].fillna(drugs_csv['BP'].mode()[0], inplace =  True)
drugs_csv['Cholesterol'].fillna(drugs_csv['Cholesterol'].mode()[0], inplace = True)
drugs_csv['Na_to_K'].fillna(drugs_csv['Na_to_K'].mean(), inplace = True)

#Encode categorical values
label_encoder = LabelEncoder()

cat_variables = drugs_csv.select_dtypes(include=['object']).columns.tolist()

for varName in cat_variables:
    drugs_csv[varName] = label_encoder.fit_transform(drugs_csv[varName])
    
def experiment( test_sizes ):
    x =  drugs_csv.drop(['Drug'],axis=1)
    y =  drugs_csv['Drug']

    accuracies = list()
    tree_nodes=list()
    accuracy_means = list()
    tree_nodes_means=list()
    
    
    for i in test_sizes:
        scores= list()
        nodes=list()
        for state in range(5):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=state)

            decision_tree=tree.DecisionTreeClassifier(criterion="entropy")
            decision_tree.fit(x_train,y_train)
            
            y_predict=decision_tree.predict(x_test)
            
            score =accuracy_score(y_test,y_predict)
            nodes.append(decision_tree.tree_.node_count)
            
            scores.append(score)
            print("State:", state+1 , "Accuracy:", score, "No. of nodes:", decision_tree.tree_.node_count)
            
        
        accuracy_means.append(sum(scores) / len(scores))
        tree_nodes_means.append(sum(nodes) / len(nodes))
        tree_nodes.append(list([min(nodes), max(nodes)]))
        accuracies.append(list([min(scores), max(scores)]))
        
    return accuracies, accuracy_means, tree_nodes, tree_nodes_means
            

print("Experiment 1 ")
accuracies, accuracy_means, tree_nodes, tree_nodes_means = experiment([0.3])
print("Best Accuracy: ",accuracies[0][1],"Tree node size: ",tree_nodes[0][1])
print("\n")

test_sizes = list([0.3,0.4,0.5,0.6,0.7])
print("Experiment 2 ")
accuracies, accuracy_means, tree_nodes, tree_nodes_means = experiment(test_sizes)

max_acc=list()
max_tree_nodes=list()



for i in range(5):
    print("Test size = ",test_sizes[i])
    print("Accuracy: ")
    print("mean: ", accuracy_means[i], "\tMin: ",accuracies[i][0], "\tMax: ",accuracies[i][1])
    print("\n")
    print("Tree size: ")
    print("mean: ", tree_nodes_means[i], "\tMin: ",tree_nodes[i][0], "\tMax: ",tree_nodes[i][1])
    print("_______________________________________________")
    max_acc.append(accuracies[i][1])
    max_tree_nodes.append(tree_nodes[i][1])
    

plt.plot(test_sizes, max_acc , marker='o')
plt.title("Plot 1")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()

plt.plot(test_sizes, max_tree_nodes , marker='o')
plt.title("Plot 2")
plt.xlabel("Training Set Size")
plt.ylabel("Number of Nodes")
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.show()
    

