from heapq import heappop, heappush
from itertools import count
import time
from tqdm import tqdm
from networkx.algorithms.shortest_paths.weighted import _weight_function

def betweenness_centrality_radius_attribute_subset(G, weight=None, radius=None, attribute=None, sources=None, targets=None):
    betweenness = dict.fromkeys(G, 0.0)
    for s in tqdm(sources, desc="Processing sources", unit="source"):
        S, P, sigma, _ = _single_source_dijkstra_path_basic_radius(G, s, weight, radius)
        betweenness, _ = _accumulate_attribute_subset(G, betweenness, S, P, sigma, s, attribute, targets)
    return betweenness

def _single_source_dijkstra_path_basic_radius(G, s, weight, radius):
    weight = _weight_function(G, weight)
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0) # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    seen = {s: 0}
    _count = count() 
    Q = [] # use Q as heap with (distance,node id) tuples
    heappush(Q, (0, next(_count), s, s)) # (distance to source, unique identifiers for nodes, pred, source)

    while Q:
        (dist, _, pred, v) = heappop(Q)
        if v in D:
            continue # already searched this node.
        sigma[v] += sigma[pred] # count path
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                if vw_dist <= radius: # add radius
                    seen[w] = vw_dist
                    heappush(Q, (vw_dist, next(_count), v, w))
                    sigma[w] = 0.0
                    P[w] = [v] # the situation of more than one shortest predecessor
            elif vw_dist == seen[w]:
                sigma[w] += sigma[v]
                P[w].append(v)

    return S, P, sigma, D

def _accumulate_attribute_subset(G, betweenness, S, P, sigma, s, attribute, targets):
    delta = dict.fromkeys(S, 0.0)
    target_set = set(targets) - {s}
    while S:
        w = S.pop()
        if w in target_set:
            attribute_w = G.nodes[w][attribute] if attribute is not None else 1
            coeff = (attribute_w + delta[w]) / sigma[w]
        else:
            coeff = delta[w] / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness, delta