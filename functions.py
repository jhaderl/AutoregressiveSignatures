import numpy as np
import esig


def time_ts(ts):
    """
    Fucntion to create a time embedding of a timeseries for calculating path signatures
    Here, we use only a leading zero with both dimensions of the timeseries as is
    Input
    ts:     a timeseries of dimension Nxp
    Output
    emb:    the embedding of the time series for which we calculate the signatures afterwards, (N+1)xd
    """

    # Create a time embedding with a leading zero
    emb = np.zeros([ts.shape[0]+1, 2])
    emb[1:,0] = ts[:,0]
    emb[1:,1] = ts[:,1]
    
    return emb


def sig_embedding(takens_emb, depth, Y):
    """
    A function to extract the path log-signatures from the data stream, i.e., the incoming timeseries embedding
    Note: To compute the full path signature, simply replace 'logsig' below with 'sig'
    Input
    takens_emb: the size of the Takens embedding (delay embedding), i.e., the length of autoregressive lags   
    depth:      the depth to which to calculate the path signature
    Y:          the time series as embedded by the function time_ts of size (N+1)xd
    Output
    sig:        the path signatures of size Nxd
    """

    N = int(Y.shape[0])
    
    if depth ==1:
        sig = np.zeros([(N-takens_emb+1), esig.logsigdim(2, depth)+1])
    else:
        sig = np.zeros([(N-takens_emb+1), esig.logsigdim(2, depth)])

    for i in range(0, N-takens_emb+1):
        snip = Y[i:i+takens_emb,:] 
        sig[i,:] = esig.stream2logsig(time_ts(snip), depth)#[[2,4,6,7,8,9]]
    
    return sig




def get_model(A, y, lamb=0):
    """
    A function to determine the coefficients of a linear model of the form Ax = b 
    This function performs a ridge regression effectively, with regularisation term lambda
    Input
    A:      a data matrix of size Nxd with N samples being rows, d features being columns
    y:      target vector of size Nxp, with N samples of dimension p 
    lamb:   the regularisation strength as a scalar value >=0
    Output
    the coefficients of the linear model, size dxp
    """

    n_col = A.shape[1]

    if lamb > 0:
        # Solve the system with numpys solver for a system of equations with Tikhonov regularisation term
        return np.linalg.solve(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y)).T
    else:
        # If lambda is 0, simply do least squares solution
        return np.linalg.lstsq(A, y)[0].T
    




def estimateAC(takens_emb, X, Y, lamb):
    """
    A function to estimate the model parameters of a differential equation of the following form:
    x(t+1) = Ax x(t) + noise
    y(t)   = C  x(t) + noise
    where x are states, y are measurements (timeseries)
    Input
    N:          the legnth of the time series
    takens_emb: the delay embedding length
    X:         a vector of states, size Nxd
    Y:         a vector of measurements, size Nxp
    lamb:       regularisation strength for the model fitting (ridge regression)
    Output
    Ax:         linear model coefficients for the state transition x(t+1) = Ax x(t)
    C:          linear model coefficients for the measurement function y(t) = C x(t)
    """

    N = int(Y.shape[0])

    # Labels for x(t)->y(t) inference, noisy Y as input
    Yn_tplus1 = Y[takens_emb:N,:]
    
    # State space at time t, x(t)
    X_t = X[0:N-takens_emb,:]
    
    # State space at time t+1, x(t+1)
    X_tplus1 = X[1:N-takens_emb+1,:]
    
    # Least squares solution for A
    #Ax = np.linalg.lstsq(X_t, X_tplus1, rcond)[0].T
    Ax = get_model(X_t, X_tplus1, lamb)
    
    # Least squares solution for C
    #C = np.linalg.lstsq(X_tplus1, Yn_tplus1, rcond)[0]
    C = get_model(X_tplus1, Yn_tplus1, lamb)
    
    return Ax, C
    


def filterfunction(Ax, C, Y, rho, lamb):
    """
    Function to reconstruct model states x from knowledge of A and C given the model assumption:
    x(t+1) = Ax x(t) + noise
    y(t)   = C  x(t) + noise
    This follows the method described in https://arxiv.org/pdf/2104.05775.pdf
    (ON THE BENEFIT OF OVERPARAMETERIZATION IN STATE RECONSTRUCTION)
    This is the computationally expensive part of the algorithm
    Input
    N:          the legnth of the time series
    Ax:         linear model coefficients for the state transition x(t+1) = Ax x(t)
    C:          linear model coefficients for the measurement function y(t) = C x(t)
    Y:          a vector of measurements, size Nxp         
    rho:        the ratio of state noise vs. measurement noise, strictly greater than zero
                if small, this leads to more denoising
                if large, this leads to a closer fit but more noise in the model A
    lamb:       the Tikhonov regularisation for the inversion of the curly O matrix
    Output
    Xhat:       Estimate of the state vector X
    Yhat:       Estimate of the measurement vector Y
    """

    N = int(Y.shape[0])
    p = int(Y.shape[1])
    n = int(Ax.shape[0])
    
    #Initialise curly C matrix
    CC = np.kron(np.identity(N), C)
    
    # Initialise curly A matrix
    AA = np.zeros(((N-1)*n, N*n))
    for k in range(0, N-1):
        AA[k*n:(k+1)*n, k*n:(k+1)*n] = -Ax
        AA[k*n:(k+1)*n, (k+1)*n:(k+1)*n+n] = np.identity(n)
    
    # Add Tikh. stabilisation
    OO=np.matmul(AA.T, AA)+rho*np.matmul(CC.T, CC) +lamb*np.identity(n*N)
    
    # Vector shape
    Y = np.reshape(Y, [p*N,1] )

    # State reconstruction
    Xhat = np.matmul(np.linalg.inv(OO), np.matmul((rho*CC.T), Y))
    
    # Denoised trajectory estimation
    Yhat = np.matmul(CC, Xhat)
    
    # Take embedding only starting at n
    Xhat = np.reshape(Xhat, [N,n])
    Yhat = np.reshape(Yhat, [N,p])
    
    return Xhat, Yhat
    