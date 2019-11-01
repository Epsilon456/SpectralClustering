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
    with open(myFile,'r') as f:
        F = f.readlines()
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
    return F


def Eigenplot(myFile,resFile=None):
    """Calculates the eigenvalues from the adjacency matrix from a given file
    Prompts the user to select two eigenvectors, then prints the graph nodes on a 
    2D plot given the eigenvectors.  The Y-axis can be set to random to allow
    a 1D plot with no overlap.  If resFile is provided, then the nodes will be labeled
    by color
    Inputs:
        myFile - The path to the text file containing the adjacency matrix
        Prompts - integers representing the index of the eigenvectors selected
        resFile(opt) - The file containing integer labels for the nodes in the graph
    Outputs:
        A list of the eigenvalues (before prompt) and a 2D plot of the nodes 
        arranged by the selected eigenvectors.
    """
    #Pull adjacency matrix from file
    A = AdjFromFile(myFile)
    #Obtain the distance diagonal of the adjacency matrix
    d = np.sum(A,axis=0)
    #Create an identity matrix
    I = np.eye(A.shape[0])
    #Calculate the Laplacian Matrix
#    L = I*d-A
    D2 = (d**-.5)*I
    L = I-np.dot(np.dot(D2,A),D2)
    #Obtain the eigenvalues and eigenvectors of the laplacian
    E,V = np.linalg.eig(L)
    
    #Print Eigenvalues
    for i in range(len(E)):
        print(i,round(E[i],5))
        
    #Select eigenvector axes
    a0 = int(input("Select first eigenvector"))
    a1 = int(input("Select second eigenvector"))
    
    #Allow you to randomize the y axis if desired
    if a1 == 'r':
        s1 = np.random.random(E.shape[0])
        print("Y axis is randomized to avoid overlap")
    else:
        s1 = V[:,a1]
 
    #Select Eigenvectors to plot
    s0 = V[:,a0]
    
    if resFile:
        F = ResFromFile(resFile)-1
        colors = ['b','r','k','y','g']
        #Plot eigenvectors and label nodes
        for i in range(len(s0)):
            color = colors[F[i]]
            plt.scatter(s0[i],s1[i],c=color)
    else:
        plt.scatter(s0,s1)
    for i in range(E.shape[0]):
        plt.text(s0[i],s1[i],str(i))
    plt.plot(s0,s1,'k',linewidth =.1)
    plt.show()
    
Eigenplot(AMatFile,resFile)
    


