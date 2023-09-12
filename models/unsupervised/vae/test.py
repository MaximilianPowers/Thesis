from sklearn.feature_selection import mutual_info_regression
import numpy as np

def compute_mig(true_latents, learned_latents):
    num_latents = true_latents.shape[1]
    mi_matrix = np.zeros((num_latents, num_latents))
    
    for i in range(num_latents):
        for j in range(num_latents):
            mi_matrix[i, j] = mutual_info_regression(true_latents[:, i].reshape(-1, 1), learned_latents[:, j])[0]
    
    mi_sorted = np.sort(mi_matrix, axis=1)
    gaps = mi_sorted[:, -1] - mi_sorted[:, -2]
    mig_score = np.mean(gaps)
    
    return mig_score
from sklearn.ensemble import RandomForestClassifier

def compute_factorvae_score(true_latents, learned_latents):
    num_latents = true_latents.shape[1]
    accuracy_scores = []
    
    for i in range(num_latents):
        classifier = RandomForestClassifier()
        classifier.fit(learned_latents, true_latents[:, i])
        accuracy = classifier.score(learned_latents, true_latents[:, i])
        accuracy_scores.append(accuracy)
        
    factorvae_score = np.mean(accuracy_scores)
    
    return factorvae_score


# Assume `vae` is your trained VAE model and `dataset` is your SineCosineDataset
true_latents = []
learned_latents = []

for i in range(len(dataset)):
    sample, latent_vars = dataset[i]
    mu, log_var = vae.encoder(torch.tensor(sample).float().unsqueeze(0))
    z = vae.reparameterize(mu, log_var)
    
    true_latents.append(latent_vars.numpy())
    learned_latents.append(z.detach().numpy())

true_latents = np.array(true_latents)
learned_latents = np.array(learned_latents)

# Compute MIG and FactorVAE Score
mig_score = compute_mig(true_latents, learned_latents)
factorvae_score = compute_factorvae_score(true_latents, learned_latents)

print("MIG Score:", mig_score)
print("FactorVAE Score:", factorvae_score)
