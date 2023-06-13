import torch.nn as nn
import torch
import numpy as np


class BioLinear(nn.Module):
    def __init__(self, in_dim, out_dim, in_fold=1, out_fold=1):
        super(BioLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_fold = in_fold
        self.out_fold = out_fold
        assert in_dim % in_fold == 0
        assert out_dim % out_fold == 0
        # compute in_cor, shape: (in_dim)
        in_dim_fold = int(in_dim/in_fold)
        out_dim_fold = int(out_dim/out_fold)
        self.in_coordinates = torch.tensor(list(np.linspace(
            1/(2*in_dim_fold), 1-1/(2*in_dim_fold), num=in_dim_fold))*in_fold, dtype=torch.float)
        self.out_coordinates = torch.tensor(list(np.linspace(
            1/(2*out_dim_fold), 1-1/(2*out_dim_fold), num=out_dim_fold))*out_fold, dtype=torch.float)
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x.clone()
        self.output = self.linear(x).clone()
        return self.output


class BioMLP(nn.Module):
    def __init__(self, prune_perc=0.1, max_prune=0.1, cc_ratio=0.5, in_dim=2, out_dim=2, w=2, depth=2, ystar=0.2, weight_factor=2, topk=10, shp=None, token_embedding=False, embedding_size=None):
        super(BioMLP, self).__init__()
        if shp == None:
            shp = [in_dim] + [w]*(depth-1) + [out_dim]
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.depth = depth

        else:
            self.in_dim = shp[0]
            self.out_dim = shp[-1]
            self.depth = len(shp) - 1

        linear_list = []
        for i in range(self.depth):
            if i == 0:
                # for modular addition
                # linear_list.append(BioLinear(shp[i], shp[i+1], in_fold=2))
                # for regression
                linear_list.append(BioLinear(shp[i], shp[i+1], in_fold=1))

            else:
                linear_list.append(BioLinear(shp[i], shp[i+1]))
        self.linears = nn.ModuleList(linear_list)

        if token_embedding == True:
            # embedding size: number of tokens * embedding dimension
            self.embedding = torch.nn.Parameter(
                torch.normal(0, 1, size=embedding_size))

        self.shp = shp
        # parameters for the bio-inspired trick
        self.l0 = ystar  # distance between two nearby layers
        self.A = weight_factor
        self.in_perm = torch.nn.Parameter(torch.tensor(
            np.arange(int(self.in_dim/self.linears[0].in_fold)), dtype=torch.float))
        # self.register_parameter(name='in_perm', param=torch.nn.Parameter(torch.tensor(np.arange(int(self.in_dim/self.linears[0].in_fold)), dtype=torch.float)))
        self.out_perm = torch.nn.Parameter(torch.tensor(
            np.arange(int(self.out_dim/self.linears[-1].out_fold)), dtype=torch.float))
        # self.register_parameter(name='out_perm', param=torch.nn.Parameter(torch.tensor(np.arange(int(self.out_dim/self.linears[-1].out_fold)), dtype=torch.float)))
        self.top_k = topk
        self.token_embedding = token_embedding
        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.original_params = None

        self.swapped = {i: [] for i in range(len(self.shp))}
        self.swappee = {i: [] for i in range(len(self.shp))}
        self.swap_self = {i: [] for i in range(len(self.shp))}

        # Fisher information
        # self.kfac = KFAC(self)
#
        # Pruning
        self.removed_nodes = {i: [] for i in range(len(shp))}
        self.count_pruned_nodes = {i: shp[i] for i in range(len(shp))}
        self.masks = [torch.ones_like(layer.linear.weight)
                      for layer in self.linears]
        self.bias_masks = [torch.ones_like(
            layer.linear.bias) for layer in self.linears]
        self.max_prune = max_prune
        self.prune = prune_perc

    def reset_swap_dict(self):
        self.swapped = {i: [] for i in range(len(self.shp))}
        self.swappee = {i: [] for i in range(len(self.shp))}
        self.swap_self = {i: [] for i in range(len(self.shp))}

    def forward(self, x):
        shp = x.shape
        in_fold = self.linears[0].in_fold
        x = x.reshape(shp[0], in_fold, int(shp[1]/in_fold))
        x = x[:, :, self.in_perm.long()]
        x = x.reshape(shp[0], shp[1])
        f = torch.nn.SiLU()
        for i in range(self.depth-1):
            x = f(self.linears[i](x))
        x = self.linears[-1](x)

        out_perm_inv = torch.zeros(self.out_dim, dtype=torch.long)
        out_perm_inv[self.out_perm.long()] = torch.arange(self.out_dim)
        x = x[:, out_perm_inv]
        # x = x[:,self.out_perm]

        return x

    def get_linear_layers(self):
        return self.linears

    def get_cc(self, bias_penalize=True, no_penalize_last=False):
        cc = 0
        num_linear = len(self.linears)
        for i in range(num_linear):
            if i == num_linear - 1 and no_penalize_last:
                weight_factor = 0.
            else:
                weight_factor = self.A
            biolinear = self.linears[i]
            dist = torch.abs(biolinear.out_coordinates.unsqueeze(
                dim=1) - biolinear.in_coordinates.unsqueeze(dim=0))
            cc += torch.sum(torch.abs(biolinear.linear.weight)
                            * (weight_factor*dist+self.l0))
            if bias_penalize == True:
                cc += torch.sum(torch.abs(biolinear.linear.bias)*(self.l0))
        if self.token_embedding:
            cc += torch.sum(torch.abs(self.embedding)*(self.l0))
        return cc

    def swap_weight(self, weights, j, k, swap_type="out"):
        with torch.no_grad():
            if swap_type == "in":
                temp = weights[:, j].clone()
                weights[:, j] = weights[:, k].clone()
                weights[:, k] = temp
            elif swap_type == "out":
                temp = weights[j].clone()
                weights[j] = weights[k].clone()
                weights[k] = temp
            else:
                raise Exception(
                    "Swap type {} is not recognized!".format(swap_type))

    def swap_bias(self, biases, j, k):

        with torch.no_grad():
            temp = biases[j].clone()
            biases[j] = biases[k].clone()
            biases[k] = temp

    def swap(self, i, j, k, permanent=False):
        # in the ith layer (of neurons), swap the jth and the kth neuron.
        # Note: n layers of weights means n+1 layers of neurons.
        # (incoming, outgoing) * weights + biases are swapped.
        if (int(j) in self.removed_nodes[i]) or (int(k) in self.removed_nodes[i]):
            return
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i == 0:
            # input layer, only has outgoing weights; update in_perm
            weights = linears[i].linear.weight
            infold = linears[i].in_fold
            fold_dim = int(weights.shape[1]/infold)
            for l in range(infold):
                self.swap_weight(weights, j+fold_dim*l, k +
                                 fold_dim*l, swap_type="in")
            # change input_perm
            self.swap_bias(self.in_perm, j, k)
        elif i == num_linear:
            # output layer, only has incoming weights and biases; update out_perm
            weights = linears[i-1].linear.weight
            biases = linears[i-1].linear.bias
            self.swap_weight(weights, j, k, swap_type="out")
            self.swap_bias(biases, j, k)
            # change output_perm
            self.swap_bias(self.out_perm, j, k)
        else:
            # middle layer : (incoming, outgoing) * weights, and biases
            weights_in = linears[i-1].linear.weight
            weights_out = linears[i].linear.weight
            biases = linears[i-1].linear.bias
            self.swap_weight(weights_in, j, k, swap_type="out")
            self.swap_weight(weights_out, j, k, swap_type="in")
            self.swap_bias(biases, j, k)

    def get_top_id(self, i):
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i == 0:
            # input layer
            weights = linears[i].linear.weight
            score = torch.sum(torch.abs(weights), dim=0)
            in_fold = linears[0].in_fold
            # print(score.shape)
            score = torch.sum(score.reshape(
                in_fold, int(score.shape[0]/in_fold)), dim=0)
        elif i == num_linear:
            # output layer
            weights = linears[i-1].linear.weight
            score = torch.sum(torch.abs(weights), dim=1)
        else:
            weights_in = linears[i-1].linear.weight
            weights_out = linears[i].linear.weight
            score = torch.sum(torch.abs(weights_out), dim=0) + \
                torch.sum(torch.abs(weights_in), dim=1)
        # print(score.shape)
        top_index = torch.flip(torch.argsort(score), [0])
        return top_index.tolist()

    def relocate_ij(self, i, j):
        # In the ith layer (of neurons), relocate the jth neuron
        if j in self.removed_nodes[i]:
            return
        linears = self.get_linear_layers()
        num_linear = len(linears)
        if i < num_linear:
            num_neuron = int(
                linears[i].linear.weight.shape[1]/linears[i].in_fold)
        else:
            num_neuron = linears[i-1].linear.weight.shape[0]
        ccs = []
        cnt_swaps = 0
        for k in range(num_neuron):
            if ((j in self.removed_nodes[i])) or (k in self.removed_nodes[i]):
                ccs.append(torch.tensor(1e10))
                continue
            self.swap(i, j, k)
            ccs.append(self.get_cc())
            self.swap(i, j, k)
            cnt_swaps += 1
        k = torch.argmin(torch.stack(ccs))
        if j != k:
            self.swapped[i].append(j)

            self.swappee[i].append(k.item())
        else:
            self.swap_self[i].append(j)

        self.swap(i, j, k, permanent=True)

    def relocate_i(self, i):
        # Relocate neurons in the ith layer
        top_id = self.get_top_id(i)
        cnt = 0
        for j in top_id:
            if cnt >= self.top_k:
                break
            if j in self.removed_nodes[i]:
                # print(f"Neuron {j} in layer {i} has been pruned")
                continue
            self.relocate_ij(i, j)
            cnt += 1

    def relocate(self):
        # Relocate neurons in the whole model
        linears = self.get_linear_layers()
        num_linear = len(linears)
        for i in range(num_linear+1):
            self.relocate_i(i)
#
    # def compute_fisher_information(self, X_val):
    #    # Initialize the FIM
    #    total_fim = None
    #    num_samples = 0
#
    #    # Iterate over the validation set
#
    #    self.kfac.register_hooks()
    #    outputs = self.forward(X_val)
    #    # Compute the FIM for the current sample
    #    self.kfac.compute_fisher()
    #    # If this is the first sample, initialize the total FIM
#
    #    if total_fim is None:
    #        total_fim = self.kfac.fisher
    #    else:
    #        # Otherwise, add the current FIM to the total FIM
    #        for key in total_fim.keys():
    #            total_fim[key] = total_fim[key] + self.kfac.fisher[key]
#
    #    # Replace the current FIM with the averaged FIM
    #    self.kfac.fisher = total_fim
#
    # def rank_nodes_by_importance(self):
    #    # Initialize a dictionary to store the importance of each node
    #    node_importance = {}
#
    #    # Iterate over the FIM
    #    n = len(self.kfac.fisher.items())
    #    for indx, (layer_name, (FIM_in, FIM_out)) in enumerate(self.kfac.fisher.items()):
    #        # Get the corresponding layer from the model
    #        layer = dict(self.named_modules())[layer_name]
#
    #        # Compute the L1 sum of weights for each neuron
#
    #        if indx == 0:
    #            l1_weights = layer.weight.abs().sum(dim=1)
#
    #            node_importance[list(self.kfac.fisher.keys())[1]] = FIM_out.sum(dim=0) * l1_weights
    #            #node_importance[list(self.kfac.fisher.keys())[1]] = l1_weights
    #        elif indx == n - 1:
    #            l1_weights = layer.weight.abs().sum(dim=0)
#
    #            node_importance[layer_name] = FIM_in.sum(dim=1) * l1_weights
    #            #node_importance[layer_name] = l1_weights
    #        else:
    #            l1_weights = layer.weight.abs().sum(dim=0)
#
    #            node_importance[layer_name] += FIM_in.sum(dim=1) * l1_weights
    #            #node_importance[layer_name] += l1_weights
#
    #            l1_weights = layer.weight.abs().sum(dim=0)
    #            node_importance[list(self.kfac.fisher.keys())[indx+1]] = FIM_out.sum(dim=0) * l1_weights
    #            #node_importance[list(self.kfac.fisher.keys())[indx+1]] = l1_weights
#
    #    # Now node_importance contains the importance of each node in each layer
    #    # We can rank the nodes by their importance
    #    ranked_nodes = {layer_name: torch.argsort(importance, descending=True)
    #                    for layer_name, importance in node_importance.items()}
#
    #    return ranked_nodes
    #
    # def prune_model(self):
    #    """
    #    Prune the model by setting the least important nodes to zero.
#
    #    Args:
    #    model: The model to be pruned.
    #    ranked_nodes: A dictionary where the keys are layer names and the values are tensors of node indices, sorted by importance.
    #    prune_ratio: The proportion of least important nodes to prune.
    #    """
    #    # Iterate over the layers
    #    cnt = 0
    #    store_weight = None
    #    ranked_nodes = self.rank_nodes_by_importance()
    #    layer_names = list(self.linears)
    #    for indx, (layer_name, node_indices) in enumerate(ranked_nodes.items()):
    #        # Get the layer from the model
    #        layer = dict(self.named_modules())[layer_name]
    #        prev_layer = layer_names[indx]
    #        # Calculate the number of nodes to prune
    #        num_nodes_to_prune = int(np.ceil(self.count_pruned_nodes[indx+1] * self.prune))
    #        if self.count_pruned_nodes[indx+1]-num_nodes_to_prune < self.shp[indx+1]*self.max_prune:
    #            continue
    #        self.count_pruned_nodes[indx+1] -= num_nodes_to_prune
    #        # Get the indices of the nodes to prune
    #        nodes_to_prune = []
    #        cnt_chosen = 0
    #        for node in node_indices:
    #            if cnt_chosen >= num_nodes_to_prune:
    #                break
    #            if node in self.removed_nodes[indx]:
    #                continue
    #            nodes_to_prune.append(node.item())
    #            cnt_chosen += 1
    #        nodes_to_prune = torch.tensor(nodes_to_prune)
    #        # Prune the nodes
    #        self.masks[indx][nodes_to_prune, :] = 0
    #        self.masks[indx+1][:, nodes_to_prune] = 0
    #
    #        prev_layer.linear.weight.data *= self.masks[indx]
    #        layer.weight.data *= self.masks[indx+1]
#
    #        # Prune the corresponding biases
    #        if indx+1 != len(self.linears) - 1:  # No bias for the last layer
    #            self.bias_masks[indx+1][nodes_to_prune] = 0
    #            layer.bias.data[nodes_to_prune] = 0
    #            self.linears[indx+1].linear.bias.data[nodes_to_prune] = 0
#
    #        for node in nodes_to_prune:
    #            self.remove_node(indx+1, int(node))
#
    #        #for key, _ in self.removed_nodes.items():
    #            #if key[0] == indx+1:
    #            #    print("-"*50)
    #            #    print(f"Max val of pruned neuron in layer {key[0]} number {key[1]}")
    #            #    print(torch.max(torch.abs(layer.weight.data[:, key[1]])))
    #            #    print(torch.max(torch.abs(self.linears[indx+1].linear.weight.data[:, key[1]])))
#

    def get_l1_sum(self):
        """
        For each layer, compute the L1 sum of weights for each neuron both off input and output weights. 
        """
        store_weight = [torch.zeros(self.shp[i+1])
                        for i in range(len(self.linears)-1)]
        for indx, linears in enumerate(self.linears):
            if indx == 0:
                store_weight[0] += torch.sum(
                    torch.abs(linears.linear.weight), dim=1)
            elif indx == len(self.linears)-1:
                store_weight[-1] += torch.sum(
                    torch.abs(linears.linear.weight), dim=0)
            else:
                store_weight[indx -
                             1] += torch.sum(torch.abs(linears.linear.weight), dim=0)
                store_weight[indx] += torch.sum(
                    torch.abs(linears.linear.weight), dim=1)
        return store_weight

    def architectural_prune(self):
        """
        Pruning mechanism based off the "cc" metric. The lower this value in proportion ot its initial state the more
        likely the model will be pruned based off the L1 sum of weights in and out. This reduces the search space for 
        swaps and relocations leading to faster convergence and better results. Use max_prune for the minimum percentage
        of nodes to be pruned in each layer. Use prune for the percentage of nodes to be pruned in each layer at each call.
        cc_ratio controls the ratio between the current, and last cc value that the model was pruned at, for this function
        to be called.
        """

        layer_names = list(self.linears)
        store_weight = self.get_l1_sum()

        for indx, layer in enumerate(list(self.linears)[1:]):
            # Get the layer from the model
            prev_layer = layer_names[indx]
            # Calculate the number of nodes to prune
            num_nodes_to_prune = int(
                np.rint(self.count_pruned_nodes[indx+1] * self.prune))
            if self.count_pruned_nodes[indx+1]-num_nodes_to_prune < self.shp[indx+1]*self.max_prune:
                continue
            self.count_pruned_nodes[indx+1] -= num_nodes_to_prune
            # Get the order of the indices of the nodes to prune
            node_indices = torch.sort(
                store_weight[indx], descending=False).indices

            nodes_to_prune = []
            cnt_chosen = 0
            for node in node_indices:
                if cnt_chosen >= num_nodes_to_prune:
                    break
                if node in self.removed_nodes[indx+1]:
                    continue
                nodes_to_prune.append(node)
                cnt_chosen += 1

            nodes_to_prune = torch.tensor(nodes_to_prune)
            if len(nodes_to_prune) == 0:
                continue
            # Prune the nodes
            self.masks[indx][nodes_to_prune, :] = 0
            self.masks[indx+1][:, nodes_to_prune] = 0

            prev_layer.linear.weight.data *= self.masks[indx]
            layer.linear.weight.data *= self.masks[indx+1]
            # Prune the corresponding biases
            if indx+1 != len(self.linears) - 1:  # No bias for the last layer
                self.bias_masks[indx+1][nodes_to_prune] = 0
                layer.linear.bias.data[nodes_to_prune] = 0
            for node in nodes_to_prune:
                self.remove_node(indx+1, int(node))

    def remove_node(self, i, j):
        # Call this method to remove a node
        self.removed_nodes[i].append(j)
