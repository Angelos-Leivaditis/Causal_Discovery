import networkx as nx

#we do not want a 0 in the denominator
def safe_div(a, b):
    return a / b if b != 0 else 0.0


def nx_sets_from_gt(GT: nx.DiGraph):
    """Ground-truth sets from a NetworkX DAG."""
    #we use a frozenset to ensure  that  u to v and v to u are recognized as the same connection 
    true_adj = {frozenset((u, v)) for (u, v) in GT.edges()} #we create a set of undirected adjacencies from the directed edges
    true_arr = {(u, v) for (u, v) in GT.edges()} #we create a set of directed arrows
    return true_adj, true_arr 


def cl_sets_from_causallearn_graph(clG, node_names):
    """
    Build adjacency + arrowhead sets from a causal-learn Graph object
    """
    M = clG.graph #  matrix that describes how nodes are connected
    p = M.shape[0] # number of variables

    pred_adj = set()  # predicted undirected adjacencies
    pred_arr = set()  # predicted directed arrows

    # Loop over all unordered pairs i, j, we chech the upper triangle og the matrix to avoid processing each pair twice
    for i in range(p):
        for j in range(i + 1, p): 
            a = M[i, j] 
            b = M[j, i]

            if a == 0 and b == 0:  # If both  are 0, there is no edge between i and j
                continue

            u, v = node_names[i], node_names[j] # indices to actual variable names
            pred_adj.add(frozenset((u, v))) #If there is a connection the names of the two variables are stored as a frozenset

            #we detect direction using the common causal-learn encoding
            if a == -1 and b == 1:
                pred_arr.add((u, v)) # u to v
            elif a == 1 and b == -1:
                pred_arr.add((v, u)) # v to u

    return pred_adj, pred_arr #all connected pairs and confirmed causal directions


def precision_recall(true_set, pred_set):
    correct = true_set & pred_set #true positives
    prec = safe_div(len(correct), len(pred_set)) # precision
    rec  = safe_div(len(correct), len(true_set)) #recall
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
