from collections import deque
import math
import random
from disjointsets import DisjointSets
from pq import PQ
from timeit import timeit

# Programming Assignment 3
# (5) After doing steps 1 through 4 below (look for relevant comments), return up here.
#     Given the output of steps 3 and 4, how does the runtime of Dijkstra's algorithm compare to Bellman-Ford?
#     Does graph density affect performance?  Does size of the graph otherwise affect performance?
#     Is Dijkstra always faster than Bellman-Ford?  If not, when is Bellman-Ford faster?


def generate_random_weighted_digraph(v,e,min_w,max_w) :
    """Generates and returns a random weighted directed graph with v vertices and e different edges.

    Keyword arguments:
    v - number of vertices
    e - number of edges
    min_w - minimum weight
    max_w - maximum weight
    """

    # ensure all vertices reachable from 0
    temp = [ x for x in range(1,v) ]
    random.shuffle(temp)
    temp.append(0)
    temp.reverse()
    edges = [ (temp[random.randrange(0,i)],temp[i]) for i in range(1,v) ]

    # if desired number of edges greater than length of current edge list, then add more edges
    if e > len(edges) :
        edgeSet = { x for x in edges }
        notYetUsedEdges = [ (x,y) for x in range(v) for y in range(v) if x != y and (x,y) not in edgeSet ]
        random.shuffle(notYetUsedEdges)
        count = e - len(edges)
        count = min(count, len(notYetUsedEdges))
        for i in range(count) :
            edges.append(notYetUsedEdges.pop())

    # generate random edge weights
    weights = [ random.randint(min_w, max_w) for x in range(len(edges)) ]

    # construct a Digraph with the lists of edges and weights generated
    G = Digraph(v, edges, weights)
    return G


def time_shortest_path_algs() :
    """Generates a table of timing results comparing two versions of Dijkstra's algorithm."""

    g1632 = generate_random_weighted_digraph(16, 32, 1, 10)
    g1660 = generate_random_weighted_digraph(16, 60, 1, 10)
    g16240 = generate_random_weighted_digraph(16, 240, 1, 10)

    g64128 = generate_random_weighted_digraph(64, 128, 1, 10)
    g64672 = generate_random_weighted_digraph(64, 672, 1, 10)
    g644032 = generate_random_weighted_digraph(64, 4032, 1, 10)

    g256512 = generate_random_weighted_digraph(256, 512, 1, 10)
    g2568160 = generate_random_weighted_digraph(256, 8160, 1, 10)
    g25665280 = generate_random_weighted_digraph(256, 65280, 1, 10)

    df = "%.7f"

    print('%0s %10s %8s %8s' % ("Algorithm", "Vertices", "Edges", "Time"))
    print('—' *43)
    print("%0s %5s %10s %15s" % ("Dijkstra ", "16", "32", df % (timeit(lambda:g1632.dijkstra(0), number=10000)/10000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "16", "32", df % (timeit(lambda:g1632.bellman_ford(0), number=10000)/10000)))
    print("%0s %5s %10s %15s" % ("Dijkstra ", "16", "60", df % (timeit(lambda:g1660.dijkstra(0), number=10000)/10000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "16", "60", df % (timeit(lambda:g1660.bellman_ford(0), number=10000)/10000)))
    print("%0s %5s %10s %15s" % ("Dijkstra ", "16", "240", df % (timeit(lambda:g16240.dijkstra(0), number=10000)/10000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "16", "240", df % (timeit(lambda:g16240.bellman_ford(0), number=10000)/10000)))

    print("%0s %5s %10s %15s" % ("Dijkstra ", "64", "128", df % (timeit(lambda:g64128.dijkstra(0), number=10000)/10000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "64", "128", df % (timeit(lambda:g64128.bellman_ford(0), number=10000)/10000)))
    print("%0s %5s %10s %15s" % ("Dijkstra ", "64", "672", df % (timeit(lambda:g64672.dijkstra(0), number=10000)/10000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "64", "672", df % (timeit(lambda:g64672.bellman_ford(0), number=10000)/10000)))
    print("%0s %5s %10s %15s" % ("Dijkstra ", "64", "4032", df % (timeit(lambda:g644032.dijkstra(0), number=1000)/1000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "64", "4032", df % (timeit(lambda:g644032.bellman_ford(0), number=1000)/1000)))

    print("%0s %5s %10s %15s" % ("Dijkstra ", "256", "6512", "%.7f" % (timeit(lambda:g256512.dijkstra(0), number=1000)/1000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "256", "6512", "%.7f" % (timeit(lambda:g256512.bellman_ford(0), number=1000)/1000)))
    print("%0s %5s %10s %15s" % ("Dijkstra ", "256", "8160", "%.7f" % (timeit(lambda:g2568160.dijkstra(0), number=1000)/1000)))
    print("%0s %6s %10s %15s" % ("Bellman ", "256", "8160", "%.7f" % (timeit(lambda:g2568160.bellman_ford(0), number=1000)/1000)))
    print("%0s %5s %10s %15s" % ("Dijkstra ", "256", "65280", "%.7f" % (timeit(lambda:g25665280.dijkstra(0), number=2)/2)))
    print("%0s %6s %10s %15s" % ("Bellman ", "256", "65280", "%.7f" % (timeit(lambda:g25665280.bellman_ford(0), number=2)/2)))

class Graph :
    """Graph represented with adjacency lists."""

    __slots__ = ['_adj']

    def __init__(self, v=10, edges=[], weights=[]) :
        """Initializes a graph with a specified number of vertices.

        Keyword arguments:
        v - number of vertices
        edges - any iterable of ordered pairs indicating the edges 
        weights - (optional) list of weights, same length as edges list
        """

        self._adj = [ _AdjacencyList() for i in range(v) ]
        i=0
        hasWeights = len(edges)==len(weights)
        for a, b in edges :
            if hasWeights :
                self.add_edge(a,b,weights[i])
                i = i + 1
            else :
                self.add_edge(a, b)

    def add_edge(self, a, b, w=None) :
        """Adds an edge to the graph.

        Keyword arguments:
        a - first end point
        b - second end point
        w - weight for the edge (optional)
        """

        self._adj[a].add(b, w)
        self._adj[b].add(a, w)

    def num_vertices(self) :
        """Gets number of vertices of graph."""
        
        return len(self._adj)

    def degree(self, vertex) :
        """Gets degree of specified vertex.

        Keyword arguments:
        vertex - integer id of vertex
        """
        
        return self._adj[vertex]._size

    def bfs(self, s) :
        """Performs a BFS of the graph from a specified starting vertex.
        Returns a list of objects, one per vertex, containing the vertex's distance
        from s in attribute d, and vertex id of its predecessor in attribute pred.

        Keyword arguments:
        s - the integer id of the starting vertex.
        """
        
        class VertexData :
            __slots__ = [ 'd', 'pred' ]

            def __init__(self) :
                self.d = math.inf
                self.pred = None

        vertices = [VertexData() for i in range(len(self._adj))]
        vertices[s].d = 0
        q = deque([s])
        while len(q) > 0 :
            u = q.popleft()
            for v in self._adj[u] :
                if vertices[v].d == math.inf :
                    vertices[v].d = vertices[u].d + 1
                    vertices[v].pred = u
                    q.append(v)
        return vertices

    def dfs(self) :
        """Performs a DFS of the graph.  Returns a list of objects, one per vertex, containing
        the vertex's discovery time (d), finish time (f), and predecessor in the depth first forest
        produced by the search (pred).
        """

        class VertexData :
            __slots__ = [ 'd', 'f', 'pred' ]

            def __init__(self) :
                self.d = 0
                self.pred = None

        vertices = [VertexData() for i in range(len(self._adj))]
        time = 0

        def dfs_visit(u) :
            nonlocal time
            nonlocal vertices

            time = time + 1
            vertices[u].d = time
            for v in self._adj[u] :
                if vertices[v].d == 0 :
                    vertices[v].pred = u
                    dfs_visit(v)
            time = time + 1
            vertices[u].f = time

        for u in range(len(vertices)) :
            if vertices[u].d == 0 :
                dfs_visit(u)
        return vertices

    def print_graph(self, with_weights=False) :
        """Prints the graph."""
        
        for v, vList in enumerate(self._adj) :
            print(v, end=" -> ")
            if with_weights :
                for u, w in vList.__iter__(True) :
                    print(u, "(" + str(w) + ")", end="\t")
            else :
                for u in vList :
                    print(u, end="\t")
            print()

    def get_edge_list(self, with_weights=False) :
        """Returns a list of the edges of the graph
        as a list of tuples.  Default is of the form
        [ (a, b), (c, d), ... ] where a, b, c, d, etc are
        vertex ids.  If with_weights is True, the generated
        list includes the weights in the following form
        [ ((a, b), w1), ((c, d), w2), ... ] where w1, w2, etc
        are the edge weights.

        Keyword arguments:
        with_weights -- True to include weights
        """
        
        edges = []
        for v, vList in enumerate(self._adj) :
            if with_weights :
                for u, w in vList.__iter__(True) :
                    edges.append(((v,u),w))
            else :
                for u in vList :
                    edges.append((v,u))
        return edges

    def mst_kruskal(self) :
        """Returns the set of edges in some
        minimum spanning tree (MST) of the graph,
        computed using Kruskal's algorithm.
        """
        
        A = set()
        forest = DisjointSets(len(self._adj))
        edges = self.get_edge_list(True)
        edges.sort(key=lambda x : x[1])
        for e, w in edges :
            if forest.find_set(e[0]) != forest.find_set(e[1]) :
                A.add(e)
                #A = A | {e}
                forest.union(e[0],e[1])
        return A

    def mst_prim(self, r=0) :
        """Returns the set of edges in some
        minimum spanning tree (MST) of the graph,
        computed using Prim's algorithm.

        Keyword arguments:
        r - vertex id to designate as the root (default is 0).
        """

        parent = [ None for x in range(len(self._adj))]
        Q = PQ()
        Q.add(r, 0)
        for u in range(len(self._adj)) :
            if u != r :
                Q.add(u, math.inf)
        while not Q.is_empty() :
            u = Q.extract_min()
            for v, w in self._adj[u].__iter__(True) :
                if Q.contains(v) and w < Q.get_priority(v) :
                    parent[v] = u
                    Q.change_priority(v, w)
        A = set()
        for v, u in enumerate(parent) :
            if u != None :
                A.add((u,v))
                #A = A | {(u,v)}
        return A


class Digraph(Graph) :

    def __init__(self, v=10, edges=[], weights=[]) :
        super(Digraph, self).__init__(v, edges, weights)

    def add_edge(self, a, b, w=None) :
        self._adj[a].add(b, w)

    def bellman_ford(self,s) :
        """Bellman Ford Algorithm for single source shortest path.

        Keyword Arguments:
        s - The source vertex.
        """

        class VertexData:
            __slots__ = ['d', 'pred']

            def __init__(self):
                self.d = math.inf
                self.pred = None

        vertices = [VertexData() for i in range(len(self._adj))]

        vertices[s].d = 0

        list_tuples = []

        def relax(u, v, w):
            if vertices[v].d > vertices[u].d + w:
                vertices[v].d = vertices[u].d + w
                vertices[v].pred = vertices[u]

        for u in range(len(vertices)):
            for v, w in self._adj[u].__iter__(True):
                relax(u, v, w)

        for u in range(len(vertices)):
            for v, w in self._adj[u].__iter__(True):
                if vertices[v].d > vertices[u].d + w:
                    return list_tuples

        for u in range(len(vertices)):
            if vertices[u].pred is not None:
                list_tuples.append((u, vertices[u].d, vertices.index(vertices[u].pred)))
            else:
                list_tuples.append((u, vertices[u].d, None))
        return list_tuples

    def dijkstra(self,s) :
        """Dijkstra's Algorithm using a binary heap as the PQ.

        Keyword Arguments:
        s - The source vertex.
        """

        class VertexData:
            __slots__ = ['d', 'pred']

            def __init__(self):
                self.d = math.inf
                self.pred = None

        vertices = [VertexData() for i in range(len(self._adj))]

        vertices[s].d = 0

        Q = PQ()
        S = []
        list = []

        vertices[s].d = 0
        Q.add(s, 0)

        for u in range(len(self._adj)):
            if u != s:
                Q.add(u, math.inf)

        while not Q.is_empty():
            u = Q.extract_min()
            S.append(u)
            for v, w in self._adj[u].__iter__(True):
                if (vertices[u].d + w) < vertices[v].d:
                    vertices[v].pred = u
                    vertices[v].d = (vertices[u].d + w)

        for v in S:
            list.append((v, vertices[v].d, vertices[v].pred))

        return list


class _AdjacencyList :

    __slots__ = [ '_first', '_last', '_size']

    def __init__(self) :
        self._first = self._last = None
        self._size = 0

    def add(self, node, w=None) :
        if self._first == None :
            self._first = self._last = _AdjListNode(node, w)
        else :
            self._last._next = _AdjListNode(node, w)
            self._last = self._last._next
        self._size = self._size + 1

    def __iter__(self, weighted=False):
        if weighted :
            return _AdjListIterWithWeights(self)
        else :
            return _AdjListIter(self)


class _AdjListNode :

    __slots__ = [ '_next', '_data', '_w' ]

    def __init__(self, data, w=None) :
        self._next = None
        self._data = data
        self._w = w


class _AdjListIter :

    __slots__ = [ '_next', '_num_calls' ]

    def __init__(self, adj_list) :
        self._next = adj_list._first
        self._num_calls = adj_list._size

    def __iter__(self) :
        return self

    def __next__(self) :
        if self._num_calls == 0 :
            raise StopIteration
        self._num_calls = self._num_calls - 1
        data = self._next._data
        self._next = self._next._next
        return data


class _AdjListIterWithWeights :

    __slots__ = [ '_next', '_num_calls' ]

    def __init__(self, adj_list) :
        self._next = adj_list._first
        self._num_calls = adj_list._size

    def __iter__(self) :
        return self

    def __next__(self) :
        if self._num_calls == 0 :
            raise StopIteration
        self._num_calls = self._num_calls - 1
        data = self._next._data
        w = self._next._w
        self._next = self._next._next
        return data, w


if __name__ == "__main__" :
        
    # here is where you will implement any code necessary to confirm that your
    # methods work correctly.
    # Code in this if block will only run if you run this module, and not if you load this module with
    # an import for use by another module.

    # (4) Call your time_shortest_path_algs() function here to output the results of step 3.

    time_shortest_path_algs()
    




    

