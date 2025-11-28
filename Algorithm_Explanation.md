# Leaf-Neighbor Greedy Vertex Cover Algorithm (LNG-VC)

## Proposed Methodology

### Overview
The proposed algorithm is a hybrid greedy approach for solving the Minimum Vertex Cover (MVC) problem. The algorithm combines two well-known heuristics: leaf-neighbor selection and highest-degree vertex selection, applied in a two-phase strategy.

### Working Mechanism

The algorithm operates in two distinct phases:

#### Phase 1: Leaf-Neighbor Removal
- The algorithm identifies all leaf vertices (vertices with degree 1) in the current graph.
- For each leaf vertex, the algorithm selects its neighbor (the only vertex it is connected to) and adds it to the vertex cover.
- The selected neighbor and all its incident edges are removed from the graph.
- This process continues iteratively until no leaf vertices remain in the graph.

**Rationale**: In a vertex cover, if a leaf vertex has degree 1, we must include either the leaf or its neighbor. Including the neighbor is always optimal because it covers the edge incident to the leaf and potentially covers other edges incident to the neighbor, making it more cost-effective.

#### Phase 2: Highest-Degree Greedy Selection
- Once no leaves remain, the algorithm switches to a greedy strategy.
- At each iteration, it identifies the vertex with the highest degree in the remaining graph.
- This vertex is added to the vertex cover and removed along with all its incident edges.
- The process continues until no edges remain in the graph.

**Rationale**: Selecting high-degree vertices maximizes the number of edges covered per vertex added, which is a proven greedy heuristic for vertex cover approximation.

### Key Features
1. **Two-Phase Strategy**: Combines the optimality of leaf-neighbor selection with the efficiency of degree-based greedy selection.
2. **Iterative Reduction**: The graph is progressively reduced, simplifying the problem at each step.
3. **Greedy Approximation**: Provides a polynomial-time approximation for the NP-hard vertex cover problem.

### Time Complexity
- **Time Complexity**: O(V + E) where V is the number of vertices and E is the number of edges
- **Space Complexity**: O(V + E) for storing the graph and maintaining data structures

---

## Pseudo Code

```
ALGORITHM: Leaf-Neighbor Greedy Vertex Cover (LNG-VC)

INPUT: Graph G = (V, E) represented as adjacency list
OUTPUT: Vertex Cover C (set of vertices)

BEGIN
    // Initialize
    C ← ∅                    // Vertex cover set
    G' ← G                   // Working copy of graph
    deg ← degree of each vertex in G'
    remaining ← V            // Set of remaining vertices
    
    // PHASE 1: Leaf-Neighbor Removal
    WHILE there exists a leaf vertex (degree = 1) in remaining DO
        leaves ← {v ∈ remaining | deg[v] = 1}
        
        FOR EACH leaf ∈ leaves DO
            IF leaf ∉ remaining OR deg[leaf] ≠ 1 THEN
                CONTINUE
            END IF
            
            // Find the neighbor of the leaf
            neighbor ← NULL
            FOR EACH adj ∈ G'[leaf] DO
                IF adj ∈ remaining THEN
                    neighbor ← adj
                    BREAK
                END IF
            END FOR
            
            IF neighbor ≠ NULL THEN
                // Add neighbor to vertex cover
                C ← C ∪ {neighbor}
                
                // Remove neighbor and all its incident edges
                FOR EACH adj ∈ G'[neighbor] DO
                    IF adj ∈ remaining THEN
                        G'[adj] ← G'[adj] \ {neighbor}
                        deg[adj] ← deg[adj] - 1
                        
                        IF deg[adj] = 0 THEN
                            remaining ← remaining \ {adj}
                        END IF
                    END IF
                END FOR
                
                remaining ← remaining \ {neighbor}
                G'[neighbor] ← ∅
                deg[neighbor] ← 0
            END IF
        END FOR
    END WHILE
    
    // PHASE 2: Highest-Degree Greedy Selection
    WHILE remaining ≠ ∅ DO
        // Check if there are remaining edges
        remaining_edges ← ∅
        FOR EACH v ∈ remaining DO
            FOR EACH u ∈ G'[v] DO
                IF u ∈ remaining AND u > v THEN
                    remaining_edges ← remaining_edges ∪ {(v, u)}
                END IF
            END FOR
        END FOR
        
        IF remaining_edges = ∅ THEN
            BREAK
        END IF
        
        // Find vertex with maximum degree
        max_degree ← -1
        max_vertex ← NULL
        
        FOR EACH v ∈ remaining DO
            IF deg[v] > max_degree THEN
                max_degree ← deg[v]
                max_vertex ← v
            END IF
        END FOR
        
        IF max_vertex = NULL THEN
            BREAK
        END IF
        
        // Add to vertex cover and remove
        C ← C ∪ {max_vertex}
        
        FOR EACH adj ∈ G'[max_vertex] DO
            IF adj ∈ remaining THEN
                G'[adj] ← G'[adj] \ {max_vertex}
                deg[adj] ← deg[adj] - 1
                
                IF deg[adj] = 0 THEN
                    remaining ← remaining \ {adj}
                END IF
            END IF
        END FOR
        
        remaining ← remaining \ {max_vertex}
        G'[max_vertex] ← ∅
        deg[max_vertex] ← 0
    END WHILE
    
    RETURN |C|  // Return size of vertex cover
END
```

---

## Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                         START                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  Initialize:                                │
        │  • C = ∅ (vertex cover set)                 │
        │  • G' = G (working copy of graph)           │
        │  • deg = degree of each vertex              │
        │  • remaining = V (set of remaining vertices)│
        └────────────────────┬────────────────────────┘
                             │
                             ▼
        ╔═══════════════════════════════════════════════╗
        ║         PHASE 1: LEAF-NEIGHBOR REMOVAL       ║
        ╚═══════════════════════════════════════════════╝
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  Find all leaves:                          │
        │  leaves = {v ∈ remaining | deg[v] = 1}    │
        └────────────────────┬────────────────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │ Leaves exist? │
                    └───────┬───────┘
                        NO │  YES
                           │       │
                           │       ▼
                           │  ┌────────────────────────────┐
                           │  │ For each leaf:             │
                           │  │ 1. Find its neighbor       │
                           │  │ 2. Add neighbor to C       │
                           │  │ 3. Remove neighbor & edges  │
                           │  │ 4. Update degrees          │
                           │  └────────────┬───────────────┘
                           │              │
                           │              └──────┐
                           │                     │
                           ▼                     │
        ╔═══════════════════════════════════════════════╗
        ║    PHASE 2: HIGHEST-DEGREE GREEDY SELECTION   ║
        ╚═══════════════════════════════════════════════╝
                           │
                           ▼
        ┌────────────────────────────────────────────┐
        │  Check remaining edges in graph            │
        └────────────────────┬────────────────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │ Edges exist?   │
                    └───────┬───────┘
                        NO │  YES
                           │       │
                           │       ▼
                           │  ┌────────────────────────────┐
                           │  │ Find vertex with max degree│
                           │  └────────────┬───────────────┘
                           │              │
                           │              ▼
                           │  ┌────────────────────────────┐
                           │  │ Add max_vertex to C        │
                           │  └────────────┬───────────────┘
                           │              │
                           │              ▼
                           │  ┌────────────────────────────┐
                           │  │ Remove max_vertex & edges  │
                           │  │ Update degrees             │
                           │  └────────────┬───────────────┘
                           │              │
                           │              └──────┐
                           │                     │
                           ▼                     │
        ┌────────────────────────────────────────────┐
        │  Return |C| (size of vertex cover)        │
        └────────────────────┬────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │                    END                     │
        └────────────────────────────────────────────┘
```

---

## Results

The algorithm was tested on a comprehensive benchmark suite. The results are computed by running `main.py`, which processes all benchmark files and updates `result.txt` with:
- **A(I)**: Vertex cover size computed by our algorithm
- **Approximation Ratio**: A(I) / OPT(I)

The approximation ratio formula is: **Approximation Ratio = A(I) / OPT(I)**

To generate results, simply run:
```bash
python3 main.py
```

This will process all benchmarks in the `Benchamarks/` directory and update `result.txt` with the computed A(I) values and approximation ratios.

