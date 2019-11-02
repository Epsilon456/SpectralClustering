# SpectralClustering
Uses spectral clustering to analyze the "Zachary's Karate Club" dataset.

# The problem
Graph theory deals with the mathematical relationship between interconnected entities.  A very famous problem which demonstrates this is "Zachary's Karate Club."  In this problem, a mathematician studied a group of students in a karate club.  In this club, a dispute arose between two of the leaders causing the club to split into two separate clubs.  By studying the interaction among students outside the club, Zachary was able to predict which students would side with each club after the split.  

Original Paper:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.336.9454&rep=rep1&type=pdf

## Graph Theory
To model this problem, each student involved was represented as a node in a graph.  If a pair of students would associate outside the karate class, this relationship was modeled as an edge connecting the two nodes.  (Note: Students who did not associate outside the class were excluded from the dataset.

Mathematically this graph can be represented as an adjacency matrix.  An adjacency matrix is constructed by listing each student along the rows of the matrix and again along the columns.  If two students interact (the two nodes are connected) the integer "1" is placed in the spot where the row of one student intersects the column of the other.  All other spots in the matrix (including the diagonal) are zeros.

### Data set
The data set consists of two text files.  The first file "Adjacency Matrix" contains the connections between the nodes in the graph and contains all the data necessary to make the predictions.  The second file, "Post split labels" contains a list of integer labels representing which of the two clubs each student associated with after the split.  These two files (included in this repository) can be found at the following link:

http://www.casos.cs.cmu.edu/tools/datasets/external/zachary/index2.html

For both scripts, the variables containing the file paths are called "AMatFile" (Adjacency Matrix) and "resFile" (labels) respectively.  You must change the paths to the paths where you have saved then before running the scripts.

## Installs
In order to run the code, you must have the numpy, scikit-learn, and matplotlib libraries installed. 

## Scripts
The repository contains two scripts.  The first, "Zachary's Karate Club.py" will read the files input and predict which of the two clubs each student will end up in after the split.  The second script, "Spectral Graphing Tool" is used for studying how nodes are placed when considering different eigenvectors.

### Zachary's Karate Club.py
This script will take in the two text files in the Data Set.  It will perform spectral clustering and group the nodes into two clusters (to represent the two clubs after the split).  A list will be printed to the console showing the index of each node, an integer representing the predicted label, and an integer representing the actual label.  Here, the number "0" represents the "blue" club and "1" represents the "red" club.

After printing this table, it will draw a plot which separates the nodes by the spectral analysis.  (The X axis is the axis associated with the smallest non-zero eigenvalue and the Y axis is the second smallest).  The plot will then label each node by the index as well as color each node depending upon the predicted labels.

### Spectral Graphing Tool
This file is not needed to run the algorithm but is useful for the curious reader.

The spectral graphing tool allows the user to make plots using different eigenvectors to study the effects.  When running, the program will print a list of eigenvalues and an index for each eigenvalue.  It will then prompt the user to type an integer to select the index of the desired eigenvalue to use for the first axis of the plot  (Be sure to press "Enter" after typing the integer).  It will then prompt the user to select the eigenvalue to use for the second axis of the plot.  

# Spectral Clustering
Spectral clustering is performed as follows assuming that the adjacency matrix, A, is a square nxn matrix.

## Step 1:Calculate the Laplacian
Calculate the Laplacian from the adjacency matrix.  This is done by the following equation:

<img src="http://latex.codecogs.com/gif.latex?L = I-DAD" border="0"/>

Where I is the nxn identiy matrix.  Both D and A are also nxn matricies which are multiplied by the dot product.

The matrix D is calculated as follows:

<img src="http://latex.codecogs.com/gif.latex?(1/sqrt(d))I" border="0"/>

Where d is a 1d array of size n consisting of the degree (number of connections) of each node.  This is calculated by summing along each row of A.

## Step 2: EigenVectors
Calculate the eigenvalues and the Eigenvectors of L.  The eigenvectors will be a nxn matrix V.  Each row corresponds to a node and each column corresponds to an eigenvalue.  Order both the list of eigenvalues and the list of eigenvectors (axis1 of V) from the smallest eigenvalue to the largest eigenvalue (remove the eigenvalue which is nearly zero).

## Step 3: Selecting Eigenvectors
After ordering the eigenvectors, take the first k eigenvectors.  (k is the number of labels which we want to assign.  For this particular problem, k=2 since the club split into 2 clubs.)  Put these eigenvectors into a nxk array called U and normalize U along axis1.

## Step 4: K-means Clustering
Perform a K-means clustering on the normalized U and form k=2 clusters.  After clustering, obtain the labels from the K-means clustering function.  The labels will tell which of the two clusters each node is associated with.  The two clusters represent the two karate classes.

## Step 5: Plot
In order to visualize the spectral clustering, plot each node on an x and y plot where x and y are the eigenvectors associated with the smallest two non-zero eigenvalues.

For further reading, look at the Spectral Graph Analysis lecture (link below) and refer to slide 51 for the [Ng et al,2001] algorithm.
http://aris.me/contents/teaching/data-mining-2016/slides/spectral.pdf


