import networkx as nx


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def nx_sets_from_gt(GT: nx.DiGraph):
    """Ground-truth sets from a NetworkX DAG."""
    true_adj = {frozenset((u, v)) for (u, v) in GT.edges()} #we build a set of undirected adjacencies from the directed edges
    true_arr = {(u, v) for (u, v) in GT.edges()} #we Build a set of directed arrows
    return true_adj, true_arr 


def cl_sets_from_causallearn_graph(clG, node_names):
    """
    Build adjacency + arrowhead sets from a causal-learn Graph object
    """
    M = clG.graph # a p x p matrix encoding with edge endpoints between variables
    p = M.shape[0] # number of variables/nodes

    pred_adj = set()  # predicted undirected adjacencies
    pred_arr = set()  # predicted directed arrows

    # Loop over all unordered pairs i, j  to avoid processing each pair twice
    for i in range(p):
        for j in range(i + 1, p):
            a = M[i, j] 
            b = M[j, i]

            if a == 0 and b == 0:  # If both  are 0, there is no edge between i and j
                continue

            u, v = node_names[i], node_names[j] # indices to actual variable names
            pred_adj.add(frozenset((u, v)))

            #we detect direction using the common causal-learn encoding
            if a == -1 and b == 1:
                pred_arr.add((u, v))
            elif a == 1 and b == -1:
                pred_arr.add((v, u))

    return pred_adj, pred_arr


def precision_recall(true_set, pred_set):
    correct = true_set & pred_set #true positives
    prec = safe_div(len(correct), len(pred_set)) # precision = (true positives) / (predicted positives)
    rec  = safe_div(len(correct), len(true_set)) #recall = (true positives) / (actual positives)
    return prec, rec, len(correct), len(pred_set), len(true_set) 


def evaluate_graph(GT: nx.DiGraph, clG, node_names, name=""):
    #Adjacency so we ignore direction
    true_adj, true_arr = nx_sets_from_gt(GT) 
    pred_adj, pred_arr = cl_sets_from_causallearn_graph(clG, node_names) 

    #Arrows 
    adj_p, adj_r, c_adj, p_adj, t_adj = precision_recall(true_adj, pred_adj)
    arr_p, arr_r, c_arr, p_arr, t_arr = precision_recall(true_arr, pred_arr)

    print(f"\n{name}")
    print(f"Adjacency precision: {adj_p:.3f}  (correct {c_adj} / predicted {p_adj})")
    print(f"Adjacency recall:    {adj_r:.3f}  (correct {c_adj} / true {t_adj})")
    print(f"Arrowhead precision: {arr_p:.3f}  (correct {c_arr} / predicted {p_arr})")
    print(f"Arrowhead recall:    {arr_r:.3f}  (correct {c_arr} / true {t_arr})")
