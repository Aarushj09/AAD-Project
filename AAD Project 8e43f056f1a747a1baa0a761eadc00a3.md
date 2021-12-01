# AAD Project

A graph is a symbolic depiction of a network and its connectivity. A graph is an ordered pair consisting of a set of vertices V and a set of edges E. Vertices can be represented as points on a plane, and these data types are known as nodes. An edge is a line that connects these nodes. Edges in a directed graph are ordered pairs, while edges in an undirected graph are unordered pairs. There are several difficulties in the actual world that may be depicted using graphs. In this project, the main aim is to explain a few of the important graph algorithms and then analyse some of them real world applications of these concepts.

# Algorithms

- Dijkstra Algorithm
    
    This is a **greedy algorithm** used to find the shortest distance between 2 points on a graph. Although there exist several algorithms that find the shortest path between 2 points, this algorithm as proposed by E.W.Dijkstra is perhaps the most efficient of them all. However, this algorithm also finds the shortest distance from the starting point to all other points. Since, this can be done with any change in the time complexity, below we discuss the algorithm and proof for getting the shortest distance from a given node to all other nodes.
    
    Algorithm
    
    In this, we are a given a graph, let `G` , with `n` nodes, some or all of which are connected by edges of given lengths. Let the total number of edges be `m` and the set of nodes be denoted by `V`.
    
    Pseudo-code
    
    Below is a pseudo-code for a function that finds the shortest distance of all nodes from `s` in `G` and stores it in `d`. Here `l(u,v)` denotes the length of the edge from `u` to `v`. If the node `v` is not reachable from `s` then, `d(v) = ∞` . Here `∞` denotes a sufficiently large number (which is guaranteed to be greater than any possible path length).
    
    ```markdown
    Dijkstra(G, s)
    	for all u ∈ V-{s}, d(u) = ∞
    	d(s) = 0
    	R = {}
    	while R != V
    		pick u !∈ R with smallest d(u)
    		R = R ∪ {u}
    		for all vertices v adjacent to u
    			if d(v) > d(u) + l(u,v)
    				d(v) = d(u) + l(u,v)
    ```
    
    This is a greedy algorithm. First we choose a vertex `u` from the set `V-R` such that `d(u)` is minimum. Then we add that vertex to `R` . Now we perform the operations called "relaxations". In each relaxation, we consider an edge from `u` to `v` and try to improve the value of `d(v)` . We do this relaxation by doing the operation `d(v) = min( d(v) , d(u)+l(u,v) )` .  
    
    Proof
    
    Now let `f(v)` denote the shortest distance from `s` to `v`. Then we need to prove that, at the end of the algorithm, `d(v) = f(v)` for all nodes reachable from `s`. We can notice that we do not update the `d` for any node in `R`. Hence we need to now prove that  `for each x ∈ R, d(x) = f(x)`
    
    We will prove this by induction.
    
    Base case : When `|R| = 1` . Since after each iteration, size of R increases by 1, the only time `|R| = 1` is when `R={s}` and since `d(s) = 0 = f(s)` , the base case is correct.
    
    Inductive Hypothesis : Let `u` be the last vertex added to `R` such that `R' = R U {u}` . Then according to our hypothesis, for each `x ∈ R', d(x) = f(x)`.
    
    Inductive Step : From the inductive hypothesis, we know that for all `x ∈ R, d(x) = f(x)`. Hence, we only need to show that `d(u) = f(u)`. We will prove this by contradiction. Let us suppose there is a path $Q$ from `s` to `u` such that `length of Q < d(u)` . Then this path must leave `R'` atleast once, because if `d` contains the shortest path by using all nodes in `R`. Consider nodes `x` and `y` such that `edge(x,y)` is the first edge in `Q` such that `x ∈ R'` and `y !∈ R'`. Then if $Q_x$ is the subpath of $Q$ from s to x, then clearly `length of Q_x + l(xy) <= length of Q`  . By our induction hypothesis, since `x ∈ R` , `d(x) = f(x)` , hence  `d(x) + l(xy) <= length of Q` . Since x already belongs to R, relaxation operation has already been performed on the edge `x,y` and hence, `d(y) <= d(x)+l(x,y)` and since, we picked `u` and not `y` to add to `R` , this means `d(u)<=d(y)` . Using these ineqaulities we get, `d(u)<d(u)` which is ofcourse false and hence a contradiction. Hence, the length of shortest path Q = d(u).
    
    ### Cpp Code
    
    We will store the graph in an adjacency list `adj`. 
    
    ```cpp
    const int INF = 1000000000;
    vector<vector<pair<int, int>>> adj;
    
    void dijkstra(int s, vector<int> & d)
    {
    	int n = adj.size();
      d.assign(n, INF);
    	d[s] = 0;
    	vector<bool> accounted(n, false);
    	for (int i = 0; i < n; i++) {
            int v = -1;
    //Extract min
            for (int j = 0; j < n; j++) {
                if (!accounted[j] && (v == -1 || d[j] < d[v]))
                    v = j;
            }
            accounted[v] = true;
    // Relaxations
            for (auto edge : adj[v]) {
                int to = edge.first;
                int len = edge.second;
    
                if (d[v] + len < d[to]) {
                    d[to] = d[v] + len;
                }
            }
        }
    }
    ```
    
    The complexity of this algorithm is $O(|V|^2 + |E|)$, which is optimal for dense graphs, where $|E|\approx|V|^2$ . However, for sparse graphs, in which $|E|<<|V|^2$, we can make use of different data structures which can reduce the time taken for `extract min` and the `relaxation` step. Some of these are:
    
    - Set : With the set data struture available in `C++` as  `set` , we can perform the `extract min` operation in $O(\log n)$ and the `relaxation` also in $O(\log n).$  Hence the final complexity with sets would be $O(n\log n+m\log n) = O(m\log n)$.
    - Heap : With the heap data struture available in `C++` as  `priority_queue` , we can perform the `extract min` operation in $O(\log m)$ and the `relaxation` also in $O(\log m).$  Hence the final complexity with sets would be $O(n\log m+m\log m) = O(m\log m)$. This change in complexity is because of the fact that `priority_queue` does not support the operation of removing an element.
    - Fibonacci Heap : With fibonacci heaps, which need to be implemented in `C++` , we can perform the `extract min` operation in $O(\log n)$ and the `relaxation` also in $O( 1).$  Hence the final complexity with sets would be $O(n\log n+m.1) = O(n\log n+m)$, which is the theoretical minimum for the shortest path search problem.
- A* search Algorithm
    
    The A* algorithm is also widely used to find the shortest distance between 2 nodes. Dijkstra’s Algorithm works well, but it wastes time exploring in directions that aren’t promising. The A* algorithm can be viewed as an extension of Dijkstra algorithm by adding a heuristic function that guides the search. A* uses the heuristic to reorder the nodes so that it’s more likely that the goal node will be encountered sooner and avoid the exploration of unnecessary vertices that do not seem to be promising. 
    
    Pseudo-code
    
    I would suggest you to go through the Dijkstra algorithm above for a better understanding, as all the notations used here are same as above, except some new variables. Here, `h` is the heuristic function, where `h(v)` estimates the length of the remaining path from `v` to `e`. The algorithm keeps an estimate of the total path cost denoted by `g(v) = d(v) + h(v)`.
    
    ```
    Astar(G,s,e)
    	for all u ∈ V-{s}, d(u) = ∞
    	d(s) = 0
    	g(s) = h(s)
    	OpenList = {}
    	CloseList = {s}
    	u = s
    	while u != e
    		// expanding
    		for all vertices v adjacent to u
    			if d(v) > d(u) + l(u,v)
    				d(v) = d(u) + l(u,v)
    			g(v) = d(v) + h(v)
    			if(v !∈ OpenList)
    				OpenList = OpenList U {v}
    		// selection
    		Select v ∈ OpenList with smallest g(v)
    		OpenList = OpenList - {v}
    		CloseList = CloseList + {v}
    		u = v 
    		if(CloseList = ∅)
    			then path not possible
    ```
    
    Similar to Dijkstra, A* works with 2 lists, the `OpenList` containing all candidate vertices and the `CloseList` that contains promising vertices. To find the most promising vertex, A* evaluates for each candidate vertex a temporary distance `g(v)` such as `g(v)=d(v)+h(v)`. It can be proved that A* finds the optimal path if for every edge `(u,v)` , the triangle inequality given by $h(u)\le l(u,v)+h(v)$ is satisfied with $h(e)=0$. The time complexity would also be similar to Dijkstra which is given by $O(n\log n + m\log n) = O(m \log n)$. 
    
    The code is almost similar to that of Dijkstra, except the part of the heuristic function, which highly depends on the problem and graph at hand and how efficient it can be. However, later in this document, we will be discussing about how we can use the Map-Reduce algorithm to improve the computation time of A* algorithm on large graphs.
    
- Kosaraju's Algorithm
    
    Kosaraju's Algorithm is an algorithm used to find strongly connected components in a directed graph. Since an undirected graph can be converted to a directed graph by replacing each with opposite edges, this algorithm can also work for undirected graphs. Strongly connected component is a maximal subset of vertices such that any two vertices of this subset are reachable from each other. 
    
    Algorithm
    
    We are given a graph `G={V,E}` which may contain loops and multiple edges. The number of vertices is `n` while the number of edges is `m`. The Kosaraju's algorithm is a DFS based algorithm. First we do a sequence of DFSs starting from each vertex and then running this DFS from each non-visited vertex. While doing this, we keep track of the entry time `tin[v]`and the exit time `tout[v]` . If `C` denotes a SCC, then $tout[C]=max(tout[v]), \forall v\in C$. Similarly, $tin[C]=min(tin[v]), \forall v\in C$. Then
    
    Theorem: Let `C` and `C′` are two different strongly connected components and there is an edge `(C,C′)` in a condensation graph between these two vertices. Then `tout[C]>tout[C′]`.
    
    - Proof
        
        There are two main different cases at the proof depending on which component will be visited by depth first search first, i.e. depending on difference between `tin[C]` and `tin[C′]`:
        
        - The component `C` was reached first. It means that depth first search comes at some vertex `v` of component `C` at some moment, but all other vertices of components `C` and `C′` were not visited yet. By condition there is an edge `(C,C′)` in a condensation graph, so not only the entire component `C` is reachable from `v` but the whole component `C′` is reachable as well. It means that depth first search that is running from vertex `v` will visit all vertices of components `C` and `C′`, so they will be descendants for `v` in a depth first search tree, i.e. $\forall u\in C\cup C', u\ne v$ we have `tout[v]>tout[u]`, as we claimed.
        - Assume that component `C′` was visited first. Similarly, depth first search comes at some vertex `v` of component `C′` at some moment, but all other vertices of components `C` and `C′` were not visited yet. But by condition there is an edge `(C,C′)` in the condensation graph, so, because of acyclic property of condensation graph, there is no back path from `C′` to `C`, i.e. depth first search from vertex `v` will not reach vertices of `C`. It means that vertices of `C` will be visited by depth first search later, so `tout[C]>tout[C′]`. This completes the proof.
    
    This theorem is the base of the algorithm. Based on this theorem, we use the following steps to find the strongly connected components.
    
    - Run sequence of depth first search of graph `G` which will return vertices with increasing exit time `tout`, i.e. some list `order`.
    - Build reverse graph `G'`, which contains all the edges of `G` in a reverse order. Run a series of depth first searches in the order determined by `order` (in reverse order). Every set of vertices, reached after the next search, will be the next strongly connected component.
    
    Pseudo-code
    
    ```
    dfs(G,order,u)
    	mark u as visited
    	for all v connected to u in G
    		if v is not visited
    			dfs(G,order,v)
    	append u to order
    
    assign(u)
    	mark u as visited
    	append u to C
    	for all v connected to u in GT
    		if v is not visited
    			assign(v)
    
    kosaraju(G)
    	GT = reverse graph of G
    	SCC = {}   // this contains the lists of SCCs
    	order = {}
    	for each u in V
    		if u is not visited before
    			dfs(G,order,u)
    	reverse(order)
    	for each u in order
    		if u not visited
    			//form a component
    			C = {}
    			assign(u)
    			append C to SCC
    ```
    
    Cpp code
    
    ```cpp
    vector<vector<int>> g, gt,scc;
    vector<bool> used;
    vector<int> order, component;
    void dfs(int v) {
        used[v] = true;
    
        for (auto u : g[v])
            if (!used[u])
                dfs(u);
    
        order.push_back(v);
    }
    
    void assign(int v) {
        used[v] = true;
        component.push_back(v);
    
        for (auto u : gt[v])
            if (!used[u])
                assign(u);
    }
    
    int main() {
        for (;;) {
            int a, b;
            // ... read next directed edge (a,b) ...
            g[a].push_back(b);
            gt[b].push_back(a);
        }
    
        used.assign(n, false);
    
        for (int i = 0; i < n; i++)
            if (!used[i])
                dfs(i);
    
        used.assign(n, false);
        reverse(order.begin(), order.end());
    
        for (auto v : order)
            if (!used[v]) {
                assign (v);
    						scc.push_back(component);
                component.clear();
            }
    }
    ```
    
    This is a linear algorithm which just performs 2 DFSs to find the SCCs. Thus the time complexity of the Kosaraju's Algorithm is $O(n+m).$
    
- Prim's Algorithm
    
    Prim's algorithm is used to the minimum spanning tree of a graph. A spanning tree is a set of edges such that any vertex can reach any other by exactly one simple path. The spanning tree with the least weight is called a minimum spanning tree. We can see that any spanning tree must necessarily contain atleast `n-1` edges. 
    
    Algorithm
    
    It is a greedy algorithm which works by gradually adding edges, one at a time. First we select and add any arbitrary vertex to the spanning tree. Then the minimum weight edge outgoing from this vertex is selected and added to the spanning tree. Now the spanning tree consists of two vertices. Now we select and add the edge with the minimum weight that has one end in the selected vertices (i.e. a vertex that is already part of the spanning tree), and the other end in the unselected vertices. We keep performing this step, i.e. every time we select and add the edge with minimal weight that connects one selected vertex with one unselected vertex. This process is repeated until the spanning tree contains all vertices, which is equivalent to saying when the spanning tree has `n-1` edges. The constructed spanning tree is guaranteed to be minimal, given the graph was originally connected.
    
    - Proof
        
        Let the graph `G` be connected. We denote by `T` the resulting graph found by Prim's algorithm, and by `S` the minimum spanning tree. We can see that `T` is indeed a spanning tree and a subgraph of `G`. We only need to show that the weights of `S` and `T` coincide.
        
        Consider the first time in the algorithm when we add an edge to `T` that is not part of `S`. Let us denote this edge with `e`, its ends by `a` and `b`, and the set of already selected vertices as `V` (`a∈V` and `b∉V`).
        
        In the minimal spanning tree `S` the vertices `a` and `b` are connected by some path `P`. On this path we can find an edge `f` such that one end of `f` lies in `V` and the other end doesn't. Since the algorithm chose `e` instead of `f`, it means that the weight of `f` is greater or equal to the weight of `e`.
        
        We add the edge `e` to the minimum spanning tree `S` and remove the edge `f`. By adding `e` we created a cycle, and since `f` was also part of the only cycle, by removing it the resulting graph is again free of cycles. And because we only removed an edge from a cycle, the resulting graph is still connected.
        
        The resulting spanning tree cannot have a larger total weight, since the weight of `e` was not larger than the weight of `f`, and it also cannot have a smaller weight since `S` was a minimum spanning tree. This means that by replacing the edge `f` with `e` we generated a different minimum spanning tree. And `e` has to have the same weight as `f`.
        
        Thus all the edges we pick in Prim's algorithm have the same weights as the edges of any minimum spanning tree, which means that Prim's algorithm really generates a minimum spanning tree.
        
    
    Pseudo-code
    
    ```
    prims(G)
    	s = 1     // s is any arbitrary starting node
    	S = {s}    
    	while S != V
    		for all edges (u,v) in E such that u ∈ S and v ∈ V-S
    			e = edge with minimum weight
    		add e to mst
    		add v corresponding to e to S
    ```
    
    Cpp code
    
    The naive code will have a time complexity of $O(n^3)$, which is really slow. In this code, we try to optimise this by storing the minimum edge to a selected vertex for each not yet selected vertex.
    
    ```cpp
    int n;
    vector<vector<int>> adj; // adjacency matrix of graph
    const int INF = 1000000000; // weight INF means there is no edge
    
    struct Edge {
        int w = INF, to = -1;
    };
    
    void prim() {
        int total_weight = 0;
        vector<bool> selected(n, false);
        vector<Edge> min_e(n);
        min_e[0].w = 0;
    
        for (int i=0; i<n; ++i) {
            int v = -1;
            for (int j = 0; j < n; ++j) {
                if (!selected[j] && (v == -1 || min_e[j].w < min_e[v].w))
                    v = j;
            }
    
            if (min_e[v].w == INF) {
                cout << "No MST!" << endl;
                exit(0);
            }
    
            selected[v] = true;
            total_weight += min_e[v].w;
            if (min_e[v].to != -1)
                cout << v << " " << min_e[v].to << endl;  // print the selected edge
    
            for (int to = 0; to < n; ++to) {
                if (adj[v][to] < min_e[to].w)
                    min_e[to] = {adj[v][to], v};
            }
        }
    
        cout << total_weight << endl;    // print the total weight of the MST
    }
    ```
    
    This algorithm can now be used to calculate the MST with a time complexity of 
    
- Floyd-Warshal Algorithm
    
    The Floyd-Warshal Algorithm is used to find the length of the shortest path between each pair of vertices in a graph. This graph may contain negative weights, but it cannot have any negative weight cycles, because it there is a negative weight cycle, we can just traverse this cycle over and over, in each iteration making the cost of the path smaller. Hence, we can make them arbitrarily small and hence the shortest path is undefined. In case of an undirected graph, no edge should have negative weight, as then we can traverse that edge back and forth and reduce the cost.
    
    Algorithm
    
    This is a dynamic programming based algorithm. We will store the distances in a matrix `d[][]`, given a graph `G` with vertices numbered from `1` to `n` . We partition this process to several incrmental phases. Before `kth` phase (k=1…n), `d[i][j]` stores the length of the shortest path from `i` to `j`, which contains only the vertices `1,2,...,k−1` as internal vertices in the path, i.e. before `kth` phase the value of `d[i][j]` is equal to the length of the shortest path from vertex `i` to the vertex `j`, if this path is allowed to enter only the vertex with numbers smaller than `k`. 
    
    Then, we will recalculate the length of all pairs `(i,j)` in the `kth` phase by
    
    $d[i][j] = min(d[i][j],d[i][k]+d[k][j])$
    
    - Proof
        
        Suppose now that we are in the `k`th phase, and we want to compute the matrix `d[][]` so that it meets the requirements for the `(k+1)th` phase. Then we have to fix the distances for some vertices pairs `(i,j)`. There are 2 possible cases:
        
        - The shortest way from `i` to `j` with internal vertices from the set `1,2,…,k` coincides with the shortest path with internal vertices from the set `1,2,…,k−1`. In this case, `d[i][j]` will not change during the transition.
        - The shortest path with internal vertices from `1,2,…,k` is shorter. This means that the new, shorter path passes through the vertex `k`. This means that we can split the shortest path between `i` and `j` into two paths: the path between `i` and `k`, and the path between `k` and `j`. It is clear that both this paths only use internal vertices of `1,2,…,k−1` and are the shortest such paths in that respect. Therefore we already have computed the lengths of those paths before, and we can compute the length of the shortest path between `i` and `j` as `d[i][k]+d[k][j]`.
    
    As a result, after the `nth` phase, the value `d[i][j]` in the distance matrix is the length of the shortest path between `i` and `j`, or is ∞ if the path between the vertices `i` and `j` does not exist.
    
    Pseudo-code
    
    ```
    FloydWarshal(G)
    	for i from 1 to n
    		for j from 1 to n
    			dist[i][j] = ∞
    	for all edges {u,v} in E
    		dist[u][v] = length of edge {u,v}
    	for k from 1 to n
    		for i from 1 to n
    			for j from 1 to n
    				dist[i][j] = min(dist[i][j],dist[i][k]+dist[k][j]
    ```
    
    Cpp code
    
    ```cpp
    int n;
    struct Edge {
        int w = INF, to = -1;
    };
    vector<vector<Edge>> adj; // adjacency matrix of graph
    const int INF = 1000000000; // weight INF means there is no edge
    
    void floydWarshal(){
    	int d[n][n]
    	for(int x=0;x<n;x++)
    		for(int y=0;y<n;y++)
    			d[x][y] = INF;
    	
    	for(int x=0;x<n;x++)
    		for(auto &y:adj[x])
    			d[x][y.to] = y.w;
    	
    	for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
            }
        }
    	}
    }
    ```
    
    The time complexity of this algorithm is obviously $O(n^3)$.
    
- Bellman-Ford Algorithm
    
    Bellman-Ford Algorithm is used to find the length of the shortest path from a given vertex to all other vertices. This is better-off than Dijkstra in case of graphs with negative edge weights, in which case Dijkstra algorithm fails to work. Although, as we will see, this has a higher time complexity than Dijkstra algorithm.
    
    Algorithm
    
    Let the weighted directed graph be `G` with `n` vertices and `m` edges, and the starting vertex be  `s`. We assume that there are no negative weight cycles in the graph. We will store the distances in `d[]`, which after execution of the algorithm will contain the answer to the problem. In the beginning, `d[v]=0`, and for all other elements `d[]` equal to infinity `∞`. This is also a dynamic programming based approach and is similar to Floyd-Warshal in some respect. The algorithm consists of several phases. Each phase scans through all edges of the graph, and the algorithm tries to produce `relaxation` along each edge `(a,b)` having weight `c`. Relaxation along the edges is an attempt to improve the value `d[b]` using value `d[a]+c`. In fact, it means that we are trying to improve the answer for this vertex using edge `(a,b)` and current response for vertex `a`.
    
    We can claim that `n-1` phases of the algorithm are sufficient to correctly calculate the lengths of all shortest paths in the graph.
    
    - Proof
        
        We will prove that after the execution of `ith` phase, the Bellman-Ford algorithm correctly finds all shortest paths whose number of edges does not exceed `i`. In other words, for any vertex `a`, let `k` denote the number of edges in the shortest path to it. Then according to this statement, the algorithm guarantees that after `kth` phase the shortest path for vertex `a` will be found. 
        
        Consider an arbitrary vertex `a` to which there is a path from the starting vertex `s`, and consider a shortest path to it $(p_0=s,p_1,...,p_k=a)$. Before the first phase, the shortest path to the vertex $p_0=s$ was found correctly. During the first phase, the edge $(p_0,p_1)$ has been checked by the algorithm, and therefore, the distance to the vertex $p_1$ was correctly calculated after the first phase. Repeating this statement `k` times, we see that after `kth` phase the distance to the vertex $p_k=a$ gets calculated correctly, which we wanted to prove.
        
        The last thing to notice is that any shortest path cannot have more than `n−1` edges. Therefore, the algorithm sufficiently goes up to the `(n−1)th` phase.
        
    
    Pseudo-code
    
    ```
    BellmanFord(G,s)
    	for all u ∈ V, d(u) = ∞
    	d(s) = 0;
    	for k from 1 to n-1
    		for all edges (u,v) in E
    			d[v] = min(d[v],d[u]+l(u,v))
    ```
    
    Cpp code
    
    Here, instead of an adjacency list, we store the graph in the form of an edge list, as it makes the processing easier.
    
    ```cpp
    struct edge
    {
        int a, b, cost;
    };
    
    int n, m, v;
    vector<edge> e;     // list of edges of graph
    const int INF = 1000000000;
    
    void solve()
    {
        vector<int> d (n, INF);
        d[v] = 0;
        for (int i=0; i<n-1; ++i)
            for (int j=0; j<m; ++j)
                if (d[e[j].a] < INF)
                    d[e[j].b] = min (d[e[j].b], d[e[j].a] + e[j].cost);
    }
    ```
    
    Clearly, we can see that the algorithm has a time complexity of $O(nm)$.
    
- Cycle detection
    
    In this algorithm, we check whether a given graph is acyclic or not. 
    
    Algorithm
    
    In this, we will perform a series of DFS in the graph. Initially all vertices are colored white. From each unvisited (white) vertex, start the DFS, mark it gray while entering and mark it black on exit. If DFS moves to a gray vertex, then we have found a cycle (if the graph is undirected, the edge to parent is not considered).
    
    Pseudo-code
    
    ```
    dfs(u)
    	color of u = gray
    	for all v adjacent to u
    		if color of v is white
    			if dfs(v) returns -1
    				return -1
    		if color of v is gray
    				return -1
    	color of u is black
    	return 0
    
    detectCycle(G)
    	color all vertices as white
    	for all v in V
    		if color of v is white
    			if dfs(v) returns -1, cycle detected
    				
    ```
    
    Cpp code
    
    In this code, we will denote white by 0, gray by 1, black by 2.
    
    ```cpp
    int n;
    vector<vector<int>> adj;
    vector<char> color;
    
    bool dfs(int v) {
        color[v] = 1;
        for (int u : adj[v]) {
            if (color[u] == 0) {
                if (dfs(u))
                    return true;
            } else if (color[u] == 1) {
                return true;
            }
        }
        color[v] = 2;
        return false;
    }
    
    void find_cycle() {
        color.assign(n, 0);
    		int flag = 0;
        for (int v = 0; v < n; v++) {
            if (color[v] == 0 && dfs(v)){
    						flag=1;
                break;
    				}
        }
    
        if (flag == 0) {
            cout << "Acyclic" << endl;
        } else {
            cout << "Cycle found" << endl;
        }
    }
    ```
    

# Applications

- Google Maps
    
    
- Friend Suggestion System
- Page Ranking Algorithm
    
    As everyone knows, it was the Google Search Engine that initiated Google’s meteoric rise into the record books. The Google Search Engine is based one simple algorithm called PageRank, which is an optimization algorithm based on a simple graph. It was first used to rank web pages in the Google search engine. The PageRank graph is generated by having all of the World Wide Web pages as nodes and any hyperlinks on the pages as edges. The edges are further characterized as weak or strong edges by weighting the edges. Pages that are linked by more credible sources such as CNN or USA.gov sites have higher weightings for the respective edges. Thus, if we compare two sites with the same number of edges, the algorithm will give the site with more links to credible sources a better rank. The total number of edges also plays a role as pages with more edges tend ****to have better ranks. Finally, when someone searches for a query, the search engine parses the strings and attempts to find the sites that closely matches the strings. It then ranks those sites according to PageRank with the best ranks appearing first. This simple procedure is how the Google Search Engine runs a search query, by making use of graphs.
    
    [](http://ilpubs.stanford.edu:8090/361/1/1998-8.pdf)
    
    Link to the original paper which details the exact original PageRank algorithm.
    
- Resource Allocation Graph
- Distributed System
    
    A distributed system's topology can be represented by a graph, with nodes representing processes and edges representing communication routes or links. As a result, distributed methods for a variety of graph theoretic issues have a wide range of applications in communication and networking. Some of the problems which can be solved by using graphs are:
    
    - Routing
    
    Routing is a critical issue in networks. The key challenge is to find and keep an acyclic path from the source of a message to its destination. The essential data is saved in a routing table, which is updated when the topology changes. When a packet arrives at a node other than its intended destination, the routing table determines how to route the packet so that it arrives at its intended destination.  A path can have many attributes. If there is a cost associated with a link, so routing via the path of least cost is required. This problem, also known as the shortest routing problem, can be reduced to a simple algorithm to find the shortest between two nodes in a graph ,which can be solved using the Dijkstra algorithm stated above, or we can use the Bellman-Ford algorithm to find the shortest path between all pairs of nodes, which can prove to be useful if we have a static topology. 
    
    - Traveral
    
    Traversal algorithms have numerous applications, starting from simple broadcast / multicast, to
    information retrieval in database, web crawling, global state collection, network routing and
    AI related problems. One such traversal problem is asynchronous message passing to all processes. In this processes send out probes(empty messages) to all of its neighbours and waits to recieve the corresponding echoes(acknowledgements). A process acts as the coordinator or initiator of activities. This problem can be reduced to finding the minimum spanning tree of the graph rooted at the given initiator node. Then the computation terminates, when a message has been sent and received through every edge exactly once. For finding these spanning trees ,we can use Prim's or Kruskal's algorithm
    
    One key thing to reaslise is that in case of distributed systems, the algorithms we use need to be robust when they work on dynamic topology, because there is usually frequent addition or deletion of nodes from a network. 
    
    There are some other important general algorithms used in distributed systems and networks like link-state algorithm and the distacne-vector routing. Distance-vector routing is a routing algorithm that is used in networking. In this, every node sends the indirect information about whatever it has learned so far about the topology to its immediate neighbors.
    
- Solving 2-SAT

## Graph Exploration Problem

## Map-Reduce Algorithm

In today's data-driven market, algorithms and apps collect data on people, processes, systems, and organisations 24 hours a day, seven days a week, resulting in massive amounts of data. The problem, however, is determining how to analyse this vast volume of data quickly and efficiently without sacrificing important insights. Here's when the MapReduce programming model comes in handy. MapReduce, which was initially utilised by Google to analyse its search results, quickly gained popularity owing to its capacity to partition and analyse terabytes of data in parallel, resulting in faster results.

MapReduce is a Hadoop programming paradigm or pattern that is used to retrieve large amounts of data stored in the Hadoop File System (HDFS). It is a critical component that is required for the Hadoop framework to function properly.

MapReduce enables concurrent processing by dividing petabytes of data into smaller chunks and processing them in parallel on Hadoop servers. Finally, it collects data from several servers and returns a consolidated result to the application.