from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function


def betweenness_centrality_bound(G, k=None, seed=None, normalized=True, weight=None, bound=None):
    '''
    This function calculates the betweenness centrality of nodes (not edges) in street networks. 
    It is adjusted based on functions in the networkx. 
    This function allows users to set bounds of betweenness centrality measurement. 
    The input graph should be an undirected weighted graph

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : int, optional (default=None)
      If k is not None use k node samples to estimate betweenness.
      The value of k <= n where n is the number of nodes in the graph.
      Higher values give better approximation.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Note that this is only used if k is not None.
    
    normalized : bool, optional
      If True the betweenness values are normalized by `2/((n-1)(n-2))`
      for graphs, where `n` is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    bound : float
      Only the pair of source and target nodes, 
      whose distance is shorter than the bound,
      are included in the betweenness centrality calculation.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.
    '''
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        S, P, sigma, _ = _single_source_dijkstra_path_basic_bounded_distance(G, s, weight, bound)
        betweenness, _ = _accumulate_basic(betweenness, S, P, sigma, s)
    # rescaling
    betweenness = _rescale(
        betweenness,
        len(G),
        normalized=normalized,
        k=k
    )
    return betweenness

def _single_source_dijkstra_path_basic_bounded_distance(G, s, weight, bound):
    weight = _weight_function(G, weight)
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    heappush(Q, (0, next(c), s, s))  # (distance to source, unique identifiers for nodes, pred, source)

    while Q:
        (dist, _, pred, v) = heappop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count path
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                if vw_dist <= bound:  # add bound
                    seen[w] = vw_dist
                    heappush(Q, (vw_dist, next(c), v, w))
                    sigma[w] = 0.0
                    P[w] = [v]  # the situation of more than one shortest predecessor
            elif vw_dist == seen[w]:
                sigma[w] += sigma[v]
                P[w].append(v)

    return S, P, sigma, D

def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness, delta

def _rescale(betweenness, n, normalized, k=None):
    if normalized:
        if n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        scale = 0.5 # this may explain why the sigma in the _single_source_dijkstra_path_basic_bounded_distance is 2 times of expacted results
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness