# PCA_Breakdown_Explanation

## I. Introduction

Principal Component Analysis (PCA) is a powerful dimensionality reduction technique widely employed in the field of data science and machine learning. It seeks to transform complex data into a lower-dimensional space while retaining as much of the original information as possible. PCA achieves this by identifying the principal components, which are linear combinations of the original features that capture the maximum variance in the data. This reduction in dimensionality not only simplifies the data but also aids in visualizing patterns and relationships, making PCA a valuable tool for exploratory data analysis, noise reduction, and enhancing the efficiency of subsequent machine learning algorithms.

 
In real-world implementations, principal components can be computed using two approaches: 1) eigendecomposition and 2) singular value decomposition (SVD). In the eigendecomposition approach, the algorithm performs eigendecomposition on the covariance matrix derived from the original dataset $X$.  This yields eigenvectors and eigenvalues, which are essentially the principal components and the information each principal component preserved. SVD has the advantage of not being restricted to square matrices like eigendecomposition. It operates directly on the original dataset $X$ without the need for computing the covariance matrix. Moreover, SVD-based matrix decomposition is considered numerically more stable, especially for large datasets. Since applying SVD directly to the data matrix is numerically more stable than performing decomposition to the covariance matrix due to numerical precision and complexity. Therefore, the popular machine learning package, sklearn, adapts the latter for PCA. However, the focus of this project is to provide a comprehensive review of PCA computations. The straightforward nature of eigendecomposition simplifies the understanding of explained variance and principal components, which is why it's the selected algorithm for this project. 



In the upcoming sections, we will walk through the process of conducting dimensional reduction using PCA. This includes: 
1) Data scaling (mean shifting), 
2) Computation of the covariance matrix, 
3) Computing the eigenvalues and eigenvectors, 
4) Sorting the eigenvectors, 
5) Projection, 
6) Evaluation. 

A concise explanation of the distinctions between eigendecomposition and SVD will also be provided. Lastly, a comparison between our model and the sklearn model will be conducted to validate the hand-coded model robustness and assess the two underlying algorithms in an example dataset.


## Reference
- [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Singular value decomposition and principal component analysis](https://link.springer.com/content/pdf/10.1007/0-306-47815-3_5.pdf)
- [Difference between SVD and eigendecomposition](https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu)

- [A plant-wide industrial process control problem](https://www.sciencedirect.com/science/article/abs/pii/009813549380018I)

- [Tennessee Eastman Process Simulation Dataset](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset/code)
