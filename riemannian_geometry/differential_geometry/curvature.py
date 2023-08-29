import numpy as np
from concurrent.futures import ThreadPoolExecutor


def batch_curvature(g, dg, ddg,tol=1e-8):
    """
    Compute Riemann curvature tensor, Ricci tensor and Ricci scalar for batch
    input.

    Parameters
    ----------

    g: np.ndarray (N, D, D)
        Diagonal metric tensor
    dg: np.ndarray (N, D, D, D)
        First derivative of the metric tensor
    ddg: np.ndarray (N, D, D, D, D)
        Second derivative of the metric tensor

    Returns
    -------
    R: np.ndarray (N, D, D, D, D)
        Riemann curvature tensor
    Ricci: np.ndarray (N, D, D)
        Ricci tensor
    Ricci_scalar: np.ndarray (N, )
        Ricci scalar
    """
    try:
        g_inv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        print(f"Singular matrix encountered. Adding {tol} to the diagonal.")

        N, D, _ = g_inv.shape
        identity = np.zeros((N, D, D))
        identity[:, np.arange(D), np.arange(D)] = tol
        g_inv = np.linalg.inv(g+identity)

    Gamma = batch_vectorised_christoffel_symbols(g_inv, dg)
    R_m, R_c, R_s = curvature(g_inv, ddg, Gamma)
    return R_m, R_c, R_s

def curvature(g_inv, ddg, Gamma):
    R_m = riemann_tensor(g_inv, ddg, Gamma)
    R_c = ricci_tensor(R_m)
    R_s = scalar_curvature(g_inv, R_c)
    return R_m, R_c, R_s

def riemann_tensor(g_inv, ddg, Gamma=None, dg=None):
    if Gamma is None:
        if dg is None:
            raise ValueError("If Gamma is not provided, dg must be provided.")
        Gamma = vectorised_christoffel_symbols(g_inv, dg)
    D = g_inv.shape[1]
    R = np.zeros((D, D, D, D))
    ddg = np.nan_to_num(ddg, nan=0.0, posinf=0.0, neginf=0.0)

    # TODO: Reduce computation time by using the symmetry of the Riemann tensor
    def compute_differential(ddg):
        return ddg + ddg.transpose(0, 1, 3, 2, 4) - ddg.transpose(0, 2, 3, 1, 4) - ddg.transpose(0, 1, 4, 2, 3)
    
    def compute_q_1(g_inv, Gamma):
        return np.einsum('Nmn, Nimj, Nnkl -> Nijkl', g_inv, Gamma, Gamma)
    
    def compute_q_2(g_inv, Gamma):
        return np.einsum('Nmn, Nilm, Nnjk -> Nijkl', g_inv, Gamma, Gamma)

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Use the executor to start the computations in parallel
        future_A1 = executor.submit(compute_differential, ddg)
        future_A2 = executor.submit(compute_q_1, g_inv, Gamma)
        future_A3 = executor.submit(compute_q_2, g_inv, Gamma)

        # Retrieve the results (this will block until each computation is done)
        differential = future_A1.result()
        Q_1 = future_A2.result()
        Q_2 = future_A3.result()
    
    R = 0.5*differential + Q_1 - Q_2
    return R
    
def ricci_tensor(R_m):
    return np.einsum('Nijkl->Nik', R_m)

def scalar_curvature(g_inv, R_c):
    return np.einsum('Nij,Nij->N', g_inv, R_c)



def vectorised_christoffel_symbols(g_inv, dg):
    """
    Get Christoffel symbols calculated from a batch of metric tensors. This
    is highly parallelised and should be faster than looping over the batch
    and calculating the Christoffel symbols for each metric tensor.

    Only use the upper diagonal part of the dg tensor since the metric tensor
    is symmetric.

    Gamma^i_{jk} = 1/2 * g^{il} * (dg_{lkj} + dg_{jlk} - dg_{klj})
    Gamma^i_{jk} = Gamma[n]^i{kj} 

    Parameters
    ----------
    g: np.ndarray (D, D)
        Diagonal metric tensor
    dg: np.ndarray (D, D, D)
        First derivative of the metric tensor

    Returns
    -------
    Gamma: np.ndarray (D, D, D)
        Christoffel symbols
    """
    store_dg = np.triu(dg) # Store only the upper diagonal part of dg since 

    # Define the functions for A1, A2, and A3
    def compute_A1(g_inv, store_dg):
        return np.einsum('il,jlk->ijk', g_inv, store_dg)

    def compute_A2(g_inv, store_dg):
        return np.einsum('il,kjl->ijk', g_inv, store_dg)

    def compute_A3(g_inv, store_dg):
        return np.einsum('il,ljk->ijk', g_inv, store_dg)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Use the executor to start the computations in parallel
        future_A1 = executor.submit(compute_A1, g_inv, store_dg)
        future_A2 = executor.submit(compute_A2, g_inv, store_dg)
        future_A3 = executor.submit(compute_A3, g_inv, store_dg)

        # Retrieve the results (this will block until each computation is done)
        A1 = future_A1.result()
        A2 = future_A2.result()
        A3 = future_A3.result()

    # Step 4: Sum A1, A2, and A3
    Gamma = (A1 + A2 - A3)*0.5

    # Verify the shape and some values of the term tensor
    return Gamma

def batch_vectorised_christoffel_symbols(g_inv, dg):
    """
    Get Christoffel symbols calculated from a batch of metric tensors. This
    is highly parallelised and should be faster than looping over the batch
    and calculating the Christoffel symbols for each metric tensor.

    Only use the upper diagonal part of the dg tensor since the metric tensor
    is symmetric.

    Gamma[n]^i_{jk} = 1/2 * g^{il} * (dg_{lkj} + dg_{jlk} - dg_{klj})
    Gamma[n]^i_{jk} = Gamma[n]^i{kj} 

    Parameters
    ----------
    g: np.ndarray (N, D, D)
        Diagonal metric tensor
    dg: np.ndarray (N, D, D, D)
        First derivative of the metric tensor

    Returns
    -------
    Gamma: np.ndarray (N, D, D, D)
        Christoffel symbols
    """
    store_dg = np.triu(dg) # Store only the upper diagonal part of dg since 

    # Define the functions for A1, A2, and A3
    def compute_A1(g_inv, store_dg):
        return np.einsum('nil,njlk->nijk', g_inv, store_dg)

    def compute_A2(g_inv, store_dg):
        return np.einsum('nil,nkjl->nijk', g_inv, store_dg)

    def compute_A3(g_inv, store_dg):
        return np.einsum('nil,nljk->nijk', g_inv, store_dg)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Use the executor to start the computations in parallel
        future_A1 = executor.submit(compute_A1, g_inv, store_dg)
        future_A2 = executor.submit(compute_A2, g_inv, store_dg)
        future_A3 = executor.submit(compute_A3, g_inv, store_dg)

        # Retrieve the results (this will block until each computation is done)
        A1 = future_A1.result()
        A2 = future_A2.result()
        A3 = future_A3.result()

    # Step 4: Sum A1, A2, and A3
    Gamma = (A1 + A2 - A3)*0.5

    # Verify the shape and some values of the term tensor
    return Gamma

# To check performance of vectorised_christoffel_symbols
#from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
#from time import process_time
#import tqdm
#import matplotlib.pyplot as plt
#
#times_1, times_2, times_3, times_4 = [], [], [], []
#
#for D in tqdm.tqdm(range(2, 32)):
#
#    dataset = np.random.randn(400, D)
#
#    manifold = LocalDiagPCA(dataset, sigma=0.5, rho=1e-3)
#
#    dataset_2 = dataset+1e-8
#    start = process_time()
#    g, dg, ddg = manifold.metric_tensor(dataset_2.transpose(), nargout=3)
#    end = process_time()
#    times_4.append(end-start)
#
#    start = process_time()
#    g_inv = np.linalg.inv(g)
#    end = process_time()
#    times_3.append(end-start)
#
#    N, D = dataset_2.shape
#    start = process_time()
#    Gamma = batch_vectorised_christoffel_symbols(g_inv, dg)
#    end = process_time()
#
#    times_1.append(end-start)
#
#    #start = process_time()
#    #Riemann = riemann_tensor(g_inv, ddg, Gamma)
#    #end = process_time()
##
#    #times_2.append(end-start)
#
#plt.plot(times_1, label="Christoffel")
#plt.plot(times_2, label="Riemann")
#plt.plot(times_3, label="g_inv")
#plt.plot(times_4, label="metric_tensor")
#plt.legend()
#plt.show()
