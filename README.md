# KNN-From-Scratch
This repo shows an implementation of the KNN ML algorithm from scratch, and verification by comparing its output to a SK-Learn KNN model both trained and evaluated on the same toy dataset. 


K-Nearest Neighbors (KNN) is a unique machine learning algorithm, because it does not have any training/optimization process, it is non probabilistic, and has a relatively small number of hyper-parameters to tune. https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm KNN can be used for both classification and regression problems. KNN is a very popular but also intuitive ML algorithm, with very basic math and code implementation. 

## How KNN works

For each sample in the test set
1. Compute the distance from current sample, to every sample in the training set. 
2. sort all the distances from least to greatest. 
3. Take the top k smallest distances, and count their labels. 
4. The most popular label is the for the current sample. 

I used the SKLearn "make_classification" method which generates synthetic classification datasets for toy examples. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html



Next: SVM + Decision Trees
