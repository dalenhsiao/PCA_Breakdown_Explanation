import numpy as np 

class myPCA():
    def __init__(
        self, 
        n_components=None # n_components is the number of dimensions preserved
        ): 
        self.n_components = n_components
        self.eVal = []
        self.pc = []
        self.explained_variance_ = 0 # explained variance is essentially eigenvalues
        self.explained_variance_ratio_ = []
        self.components_ = []

    # Compute the covariance matrix
    def covMat(self,X):
        # or return (X.T @ X)/X.shape[0] (X is centered)
        S = np.cov(X, rowvar=False)
        return S
    
    # perform eigen decomposition
    def eigenDecomposition(self,A): 
        eVal, eVec = np.linalg.eig(A)
        return eVal, eVec
    
    # Generate PCs and eigenvalues with PCs sorted by the weights of corresponding eigenvalues
    def getPCs(self,cov):

        eVal, eVec = self.eigenDecomposition(cov)
        indices = np.argsort(-eVal)
        e = eVal[indices,]
        pc = -eVec[:,indices] # we can flip the direction of eigenvectors 
        return pc, e
    
    
    # Fit function to generate estimator (model) 
    def fit(self, X):
        # compute covariance matrix 
        cov = self.covMat(X)

        # perform eigen decomposition to obtain eigenvalues and eigenvectors(PCs)
        self.pc, self.eVal = self.getPCs(cov)
        
        # if user specified n_component is None, n_components = min(n_samples, n_features)
        if not self.n_components:
            self.n_components == min(X.shape[0], X.shape[1])

        # presevered components
        self.components_= self.pc[:, :self.n_components].T

        # explained variance is essentially eigenvalues
        self.explained_variance_ = np.array([*self.eVal[:self.n_components]])
        # compute explained variance ratio
        self.explained_variance_ratio_ = self.get_explained_variance_ratio_()

        self.model = np.array(self.pc) # model is the PCs matrix, where each entry of PC is the loading of each feature's contribution to the PC

        print(myPCA)

    
    # model transform 
    def transform(self,X):
        try:
            # extract first N PCs 
            T = X @ self.model[:, :self.n_components]
            return T 
        except AttributeError: # if not perform fit before transforming 
            print("PCA model has to fit before transforming, otherwise, use `fit_transform`")
        
    
    # Combining fit and trasnform function
    def fit_transform(self,X): # in sklearn: fit_transform
        self.fit(X) # call fit 
        T = self.transform(X) # transform the data onto new projections
        return T
    
        
    # explained variance ratio
    def get_explained_variance_ratio_(self): # in Sklearn: explained_variance_ratio_
        self.allVar = sum(self.eVal) # total vairance explained
        explained = [(val/self.allVar) for val in self.eVal] # ratio of explained variance for each PC
        return explained[:self.n_components]
        
    # cumulative explained ratio
    def get_cumulative_explained_ratio(self):
        cumulative_explained = [] # cumulative explained variance
        cum = 0
        for ex in self.explained_variance_ratio_:
            cum += ex
            cumulative_explained.append(cum)
        return np.array(cumulative_explained)