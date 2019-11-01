AMatFile = r"D:\Kaggle Datasets\Zacharys Karate Club\Adjacency Matrix.txt"
resFile = r"D:\Kaggle Datasets\Zacharys Karate Club\Post split labels.txt"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

######################Load AMat#######################################################
def AdjFromFile(myFile):
    """Loads an adjacency matrix from a text file
    Inputs: file path
    Outputs: Adjacency matrix
    """
    #Reads the text file
    with open(myFile,'r') as f:
        F = f.readlines()
    #Convert the matrix in the text (space separated) to a 2d numpy array
    A = []
    for line in F:
        a = line.split()
        A.append(a)
    A = np.array(A).astype(np.int32)
    return A

def ResFromFile(resFile):
    """Outputs label vector by reading results file
    """
    with open(resFile,'r') as f:
        F = f.readlines()
    F = np.array(F).astype(np.int)
    return F-1


def Eigen(A):
    """Calculates the the eigenvalues and eigenvectors associated with 
    the normalized Laplacian of a given adjacency matrix.
    Inputs:
        A - Adjacecny matrix (must be a square matrix)
    Outputs:
        E - An array of eigenvalues
        V - an array of eigenvectors 
            axis0 - each node from the given adjacency matrix
            axis1 - each dimension of the eigenspace occupied by the eigenvector
    """
    #Obtain the distance diagonal of the adjacency matrix
    d = np.sum(A,axis=0)
    #Create an identity matrix
    I = np.eye(A.shape[0])
    #Calculate the Laplacian Matrix
    D2 = (d**-.5)*I
    L = I-np.dot(np.dot(D2,A),D2)

    #Obtain the eigenvalues and eigenvectors of the laplacian
    E,V = np.linalg.eig(L)
    return E,V
    

def Eigenplot(A,labels=None):
    """Calculates the eigenvalues from a given adjacency matrix. Plots the nodes
    of the graph on a 2d plot where each axis corresponds to the two eigenvectors with
    the two smallest, nonzero eigenvales.
    
    Inputs:
        A - The square adjacency matrix
        results(optional) - A 1d array consisting of integer labels for each node in the graph.
            Labels must contain fewer than 5 labels to properly color the plot
    Outputs:
        2D plot of the nodes 
        arranged by the selected eigenvectors.
    """
    #Calculate the eigenvalues and eigenvectors of the normalized laplacian given the adjacency matrix.
    E,V = Eigen(A)

    #Sort the eigenvaleus from smallest to largest (excluding the smallest since the smallest is nearly zero.)        
    indices = np.argsort(E)[1:]

    #Select the indices of the two smallest eigenvalues (excluding 0)
    a0 = indices[0]
    a1 = indices[1]
    
    #Select Eigenvectors to plot
    s0 = V[:,a0]
    s1 = V[:,a1]
    
    #If an array consisting of the true labels is provided, color each node by the label.
    if labels is not None:
        #Supports 5 different colors for 5 different labels.
        colors = ['b','r','k','y','g']
        #Plot eigenvectors and label nodes different colors depending upon their labels
        for i in range(len(s0)):
            color = colors[F[i]]
            plt.scatter(s0[i],s1[i],c=color)
    #If no labels, then plot all nodes the same color
    else:
        plt.scatter(s0,s1)
    #Label each node with its index
    for i in range(E.shape[0]):
        plt.text(s0[i],s1[i],str(i))
    #Plot the graph edges
    plt.plot(s0,s1,'k',linewidth =.1)
    plt.show()
    
    
def Cluster(E,V,k):
    """Run the normalized spectral clustering algorithm to assign labels to the nodes on the graph.
    Inputs:
        E - The Eigenvalues of the normalized Laplacian
        V - The Eigenvectors assiciated with E
        k - The number of possible classes to label.
    Outputs:
        result - 1d numpy array consisint of integer labels for each node.
    """
    #Obtain the indices of the eigenvalues in order from smallest to largest (excluding the smallest 
        #eigenvalue which is zero.)
    indices = np.argsort(E)[1:]
    #Create a list of the smallest eigenvectors.
    U = []
    for i in range(k):
        index = indices[i]
        #Note that axis0 corresponds to nodes and axis1 corresponds to eigenvalues.
        U.append(V[:,index])
    #Transpose U so that it is of shape [nodes,k]
    U = np.array(U).T
    #Normalize each row of U
    norms = np.expand_dims(np.linalg.norm(U,axis=1),-1)
    Un = U/norms
    
    #Calculate the K means clustering of the normalized Un. This will group all nodes into k clusters
        #Each cluster will be assigned a label.
    km = KMeans(n_clusters=k).fit(Un)
    result = km.labels_
    #Return the array of labels.
    return result
        
#Obtain the Adjacency matrix from the file
A = AdjFromFile(AMatFile)
#Obtain the Labels from the other file.
F = ResFromFile(resFile)
#Calculate the eigenvalues and eigenvectors of the Laplacian of A.
E,V = Eigen(A)
#Run the spectral clustering algorithm.
Result = Cluster(E,V,2)

#Print the actual and predicted labels for each node.
print("Node","Actual","Prediction")
for i in range(len(F)):
    print(i,"\t",F[i],Result[i])

#Plot the results of the graph
Eigenplot(A,Result)




