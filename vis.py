#!/usr/bin/env python3
"""
Step-by-step visualization of the Leaf-Neighbor Greedy Vertex Cover Algorithm
Shows how the algorithm works on a small example graph
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
from collections import defaultdict
import time
import os
import sys
import glob

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Color scheme
COLOR_VERTEX_NORMAL = '#4A90E2'      # Blue
COLOR_VERTEX_LEAF = '#F5A623'        # Orange (leaf)
COLOR_VERTEX_SELECTED = '#7ED321'    # Green (in vertex cover)
COLOR_VERTEX_REMOVED = '#D3D3D3'     # Gray (removed)
COLOR_EDGE_NORMAL = '#9B9B9B'        # Gray
COLOR_EDGE_COVERED = '#50C878'       # Green (covered by vertex cover)
COLOR_EDGE_REMOVED = '#FF6B6B'       # Red (removed)


def read_graph_from_txt(filepath):
    """Read graph from .txt file (edge list format)"""
    edges = []
    vertices = set()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:  # Skip self-loops
                    edges.append((u, v))
                    vertices.add(u)
                    vertices.add(v)
    
    return edges, vertices


def read_graph_from_xlsx(filepath):
    """Read graph from .xlsx file (edge list format)"""
    if not HAS_PANDAS:
        raise ImportError("pandas is required to read .xlsx files")
    
    df = pd.read_excel(filepath, header=None)
    edges = []
    vertices = set()
    
    for _, row in df.iterrows():
        if len(row) >= 2 and pd.notna(row[0]) and pd.notna(row[1]):
            u, v = int(row[0]), int(row[1])
            if u != v:  # Skip self-loops
                edges.append((u, v))
                vertices.add(u)
                vertices.add(v)
    
    return edges, vertices


def load_benchmark_graph(filepath=None, benchmark_name=None, use_example=False):
    """
    Load a graph from a benchmark file or use default example
    If filepath is provided, load from that file
    If benchmark_name is provided, try to find matching file
    If use_example=True, use small example graph
    Otherwise, use default benchmark (graph50-06)
    """
    if filepath and os.path.exists(filepath):
        # Load from provided filepath
        if filepath.endswith('.txt'):
            edges, vertices = read_graph_from_txt(filepath)
        elif filepath.endswith('.xlsx'):
            edges, vertices = read_graph_from_xlsx(filepath)
        else:
            print(f"Unsupported file type: {filepath}")
            return load_default_benchmark()
        
        G = nx.Graph()
        G.add_edges_from(edges)
        print(f"Loaded benchmark: {os.path.basename(filepath)}")
        print(f"  Vertices: {len(vertices)}, Edges: {len(edges)}")
        return G
    
    elif benchmark_name:
        # Try to find benchmark file
        benchmark_dir = os.path.join(os.path.dirname(__file__), 'Benchamarks-20251129', 'Benchamarks')
        if os.path.exists(benchmark_dir):
            # Search for matching file
            txt_files = glob.glob(os.path.join(benchmark_dir, f'*{benchmark_name}*.txt'))
            xlsx_files = glob.glob(os.path.join(benchmark_dir, f'*{benchmark_name}*.xlsx'))
            all_files = txt_files + xlsx_files
            
            if all_files:
                filepath = all_files[0]
                if filepath.endswith('.txt'):
                    edges, vertices = read_graph_from_txt(filepath)
                else:
                    edges, vertices = read_graph_from_xlsx(filepath)
                
                G = nx.Graph()
                G.add_edges_from(edges)
                print(f"Loaded benchmark: {os.path.basename(filepath)}")
                print(f"  Vertices: {len(vertices)}, Edges: {len(edges)}")
                return G
    
    if use_example:
        # Use small example graph
        print("Using simple example graph (18 vertices)")
        return create_example_graph()
    
    # Default: use real benchmark
    return load_default_benchmark()


def load_default_benchmark():
    """Load default benchmark (graph50-06)"""
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'Benchamarks-20251129', 'Benchamarks')
    default_file = os.path.join(benchmark_dir, 'graph50-06.xlsx')
    
    if os.path.exists(default_file):
        edges, vertices = read_graph_from_xlsx(default_file)
        G = nx.Graph()
        G.add_edges_from(edges)
        print(f"Using default benchmark: graph50-06.xlsx")
        print(f"  Vertices: {len(vertices)}, Edges: {len(edges)}")
        return G
    else:
        # Fallback to example if benchmark not found
        print("Default benchmark not found, using simple example graph")
        return create_example_graph()


def create_example_graph():
    """
    Create a small example graph for visualization
    This graph has both leaves and high-degree vertices
    """
    # Create a graph with leaves and cycles
    G = nx.Graph()
    
    # Add edges to create an interesting structure
    edges = [
        (1, 2), (1, 3),           # Vertex 1 has degree 2
        (2, 4), (2, 5),           # Vertex 2 has degree 3
        (3, 6),                   # Vertex 3 has degree 2
        (4, 7),                   # Vertex 4 has degree 2 (leaf initially)
        (5, 8),                   # Vertex 5 has degree 2
        (6, 9),                   # Vertex 6 has degree 2
        (7, 10),                  # Vertex 7 is a leaf (degree 1)
        (8, 9),                   # Creates a cycle
        (9, 10),                  # Connects to leaf
        (10, 11),                 # Vertex 10 has degree 3
        (11, 12),                 # Vertex 11 has degree 2
        (12, 13),                 # Vertex 12 has degree 2
        (13, 14),                 # Vertex 13 is a leaf (degree 1)
        (14, 15),                 # Vertex 14 has degree 2
        (15, 16),                 # Vertex 15 has degree 2
        (16, 17),                 # Vertex 16 has degree 2
        (17, 18),                 # Vertex 17 is a leaf (degree 1)
    ]
    
    G.add_edges_from(edges)
    return G


def build_graph_from_edges(edges):
    """Build graph representation from edges"""
    graph = defaultdict(set)
    degree = defaultdict(int)
    
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        degree[u] += 1
        degree[v] += 1
    
    return graph, degree


def greedy_vertex_cover_with_steps(graph, degree):
    """
    Greedy Vertex Cover Algorithm with step-by-step tracking
    Returns list of steps for visualization
    """
    # Create copies
    g = {v: set(neighbors) for v, neighbors in graph.items()}
    deg = degree.copy()
    
    vertex_cover = set()
    remaining_vertices = set(g.keys())
    steps = []
    
    # Phase 1: Remove neighbors of leaves
    phase = 1
    iteration = 0
    
    while True:
        iteration += 1
        leaves = [v for v in remaining_vertices if deg[v] == 1]
        
        if not leaves:
            break
        
        # For each leaf, remove its neighbor
        for leaf in leaves:
            if leaf not in remaining_vertices or deg[leaf] != 1:
                continue
            
            # Find the neighbor
            neighbor = None
            for n in g[leaf]:
                if n in remaining_vertices:
                    neighbor = n
                    break
            
            if neighbor is not None:
                # Record step
                step = {
                    'phase': phase,
                    'iteration': iteration,
                    'action': 'leaf_neighbor',
                    'leaf': leaf,
                    'selected': neighbor,
                    'vertex_cover': vertex_cover.copy(),
                    'remaining_vertices': remaining_vertices.copy(),
                    'graph_state': {v: set(neighbors) for v, neighbors in g.items()},
                    'degrees': deg.copy()
                }
                steps.append(step)
                
                # Add neighbor to vertex cover
                vertex_cover.add(neighbor)
                
                # Remove neighbor and all its edges
                for adj in list(g[neighbor]):
                    if adj in remaining_vertices:
                        g[adj].discard(neighbor)
                        deg[adj] -= 1
                        if deg[adj] == 0:
                            remaining_vertices.discard(adj)
                
                remaining_vertices.discard(neighbor)
                g[neighbor].clear()
                deg[neighbor] = 0
    
    # Phase 2: Remove highest-degree nodes
    phase = 2
    iteration = 0
    
    while remaining_vertices:
        iteration += 1
        
        # Find all remaining edges
        remaining_edges = []
        for v in remaining_vertices:
            for u in g[v]:
                if u in remaining_vertices and u > v:
                    remaining_edges.append((v, u))
        
        if not remaining_edges:
            break
        
        # Find vertex with highest degree
        max_degree = -1
        max_vertex = None
        
        for v in remaining_vertices:
            if deg[v] > max_degree:
                max_degree = deg[v]
                max_vertex = v
        
        if max_vertex is None:
            break
        
        # Record step
        step = {
            'phase': phase,
            'iteration': iteration,
            'action': 'highest_degree',
            'selected': max_vertex,
            'degree': max_degree,
            'vertex_cover': vertex_cover.copy(),
            'remaining_vertices': remaining_vertices.copy(),
            'graph_state': {v: set(neighbors) for v, neighbors in g.items()},
            'degrees': deg.copy()
        }
        steps.append(step)
        
        # Add to vertex cover and remove
        vertex_cover.add(max_vertex)
        
        for adj in list(g[max_vertex]):
            if adj in remaining_vertices:
                g[adj].discard(max_vertex)
                deg[adj] -= 1
                if deg[adj] == 0:
                    remaining_vertices.discard(adj)
        
        remaining_vertices.discard(max_vertex)
        g[max_vertex].clear()
        deg[max_vertex] = 0
    
    # Final step
    steps.append({
        'phase': 3,
        'iteration': 0,
        'action': 'complete',
        'vertex_cover': vertex_cover.copy(),
        'remaining_vertices': set(),
        'graph_state': {v: set(neighbors) for v, neighbors in g.items()},
        'degrees': deg.copy()
    })
    
    return steps, vertex_cover


def visualize_step(step, ax, pos, G_original):
    """Visualize a single step of the algorithm"""
    ax.clear()
    
    graph_state = step['graph_state']
    remaining_vertices = step['remaining_vertices']
    vertex_cover = step['vertex_cover']
    
    # Create NetworkX graph from current state - ONLY remaining vertices
    G_current = nx.Graph()
    for v, neighbors in graph_state.items():
        if v in remaining_vertices:
            for n in neighbors:
                if n in remaining_vertices:
                    G_current.add_edge(v, n)
    
    # Add isolated vertices that are still remaining
    for v in remaining_vertices:
        if v not in G_current:
            G_current.add_node(v)
    
    # Only visualize remaining vertices (removed nodes are not shown)
    if len(G_current.nodes()) == 0:
        ax.text(0.5, 0.5, 'All vertices removed\nAlgorithm complete', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Create position for remaining vertices only
    pos_current = {v: pos[v] for v in G_current.nodes() if v in pos}
    
    # Determine vertex colors (only for remaining vertices)
    node_colors = []
    for node in G_current.nodes():
        if node in vertex_cover:
            node_colors.append(COLOR_VERTEX_SELECTED)
        elif step['action'] == 'leaf_neighbor' and node == step.get('leaf'):
            node_colors.append(COLOR_VERTEX_LEAF)
        elif step['action'] in ['leaf_neighbor', 'highest_degree'] and node == step.get('selected'):
            node_colors.append(COLOR_VERTEX_SELECTED)
        else:
            node_colors.append(COLOR_VERTEX_NORMAL)
    
    # Determine edge colors (only for remaining edges)
    edge_colors = []
    for edge in G_current.edges():
        u, v = edge
        if u in vertex_cover or v in vertex_cover:
            edge_colors.append(COLOR_EDGE_COVERED)
        else:
            edge_colors.append(COLOR_EDGE_NORMAL)
    
    # Draw graph - only remaining vertices and edges
    nx.draw_networkx_nodes(G_current, pos_current, node_color=node_colors, 
                          node_size=800, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G_current, edgelist=list(G_current.edges()),
                          pos=pos_current, edge_color=edge_colors, width=2, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(G_current, pos_current, ax=ax, font_size=10, font_weight='bold')
    
    # Add title with step information
    phase_names = {1: "Phase 1: Leaf-Neighbor Removal", 
                   2: "Phase 2: Highest-Degree Greedy",
                   3: "Complete"}
    
    title = f"{phase_names.get(step['phase'], 'Algorithm')}"
    if step['action'] == 'leaf_neighbor':
        title += f"\nIteration {step['iteration']}: Leaf {step['leaf']} → Select neighbor {step['selected']}"
    elif step['action'] == 'highest_degree':
        title += f"\nIteration {step['iteration']}: Select vertex {step['selected']} (degree {step['degree']})"
    elif step['action'] == 'complete':
        title += f"\nVertex Cover Size: {len(vertex_cover)}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend (removed nodes/edges are not shown, so remove from legend)
    legend_elements = [
        mpatches.Patch(color=COLOR_VERTEX_NORMAL, label='Normal Vertex'),
        mpatches.Patch(color=COLOR_VERTEX_LEAF, label='Leaf (degree 1)'),
        mpatches.Patch(color=COLOR_VERTEX_SELECTED, label='In Vertex Cover'),
        mpatches.Patch(color=COLOR_EDGE_COVERED, label='Covered Edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)


def create_animation(filepath=None, benchmark_name=None, use_example=False, max_vertices=100):
    """Create animated visualization"""
    print("Creating step-by-step visualization...")
    
    # Load graph (from benchmark or default)
    G = load_benchmark_graph(filepath=filepath, benchmark_name=benchmark_name, use_example=use_example)
    
    # Check graph size
    n = len(G.nodes())
    if n > max_vertices:
        print(f"\n⚠ Warning: Graph has {n} vertices (recommended max: {max_vertices})")
        print(f"   Visualization may be slow for large graphs.")
        response = input("   Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("   Aborted.")
            return None, [], set()
    
    # Get edges and build graph representation
    edges = list(G.edges())
    graph, degree = build_graph_from_edges(edges)
    
    # Run algorithm with step tracking
    steps, final_cover = greedy_vertex_cover_with_steps(graph, degree)
    
    print(f"Algorithm completed in {len(steps)} steps")
    print(f"Final vertex cover size: {len(final_cover)}")
    print(f"Vertices in cover: {sorted(final_cover)}")
    
    # Check if display is available
    import os
    import matplotlib
    # Check if we're in a non-interactive backend
    backend = matplotlib.get_backend()
    has_display = (os.environ.get('DISPLAY') is not None and 
                   backend.lower() not in ['agg', 'pdf', 'svg', 'ps'])
    
    if not has_display:
        print("\n⚠ No interactive display detected. Saving static images instead...")
        print("   (Use --static flag to always save images)")
        # Save static images instead
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        for i, step in enumerate(steps):
            fig, ax = plt.subplots(figsize=(14, 10))
            visualize_step(step, ax, pos, G)
            plt.tight_layout()
            filename = f"step_{i+1:02d}_phase{step['phase']}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close()
        print(f"\n✓ Saved {len(steps)} step visualizations")
        return None, steps, final_cover
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Create animation
    def animate(frame):
        if frame < len(steps):
            visualize_step(steps[frame], ax, pos, G)
        else:
            # Show final state
            visualize_step(steps[-1], ax, pos, G)
    
    # Create animation (slower for better viewing)
    anim = FuncAnimation(fig, animate, frames=len(steps), 
                        interval=2000, repeat=True, blit=False)
    
    plt.tight_layout()
    
    # Try to show, but if it fails, save images instead
    try:
        # Check backend before trying to show
        if matplotlib.get_backend().lower() in ['agg', 'pdf', 'svg', 'ps']:
            raise RuntimeError("Non-interactive backend")
        plt.show(block=False)
        print("\n✓ Animation window opened (close window to continue)")
    except Exception as e:
        print(f"\n⚠ Could not display interactively: {e}")
        print("   Saving static images instead...")
        plt.close(fig)
        # Save static images instead
        output_folder = "benchmark_steps_animation"
        os.makedirs(output_folder, exist_ok=True)
        for i, step in enumerate(steps):
            fig2, ax2 = plt.subplots(figsize=(14, 10))
            visualize_step(step, ax2, pos, G)
            plt.tight_layout()
            filename = os.path.join(output_folder, f"step_{i+1:02d}_phase{step['phase']}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close(fig2)
        print(f"\n✓ Saved {len(steps)} step visualizations to {output_folder}/")
        return None, steps, final_cover
    
    return anim, steps, final_cover


def create_static_visualization(filepath=None, benchmark_name=None, use_example=False, max_vertices=100):
    """Create static step-by-step visualization (saves images)"""
    print("Creating static step-by-step visualization...")
    
    # Load graph (from benchmark or default)
    G = load_benchmark_graph(filepath=filepath, benchmark_name=benchmark_name, use_example=use_example)
    
    # Check graph size - warn if too large
    n = len(G.nodes())
    if n > max_vertices:
        print(f"\n⚠ Warning: Graph has {n} vertices (recommended max: {max_vertices})")
        print(f"   Visualization may be slow or unclear for large graphs.")
        print(f"   Consider using a smaller benchmark or subgraph.")
        response = input("   Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("   Aborted.")
            return
    
    # Get edges and build graph representation
    edges = list(G.edges())
    graph, degree = build_graph_from_edges(edges)
    
    # Run algorithm with step tracking
    steps, final_cover = greedy_vertex_cover_with_steps(graph, degree)
    
    print(f"Algorithm completed in {len(steps)} steps")
    print(f"Final vertex cover size: {len(final_cover)}")
    print(f"Vertices in cover: {sorted(final_cover)}")
    
    # Determine output folder
    if use_example:
        # Simple example goes to simple_example_steps (don't change this)
        output_folder = "simple_example_steps"
    else:
        # Benchmarks go to benchmark_steps folder
        output_folder = "benchmark_steps"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving steps to: {output_folder}/")
    
    # Use spring layout (adjust k based on graph size)
    k_value = min(1.5, max(0.5, 1.5 * (100 / n)))  # Scale k inversely with size
    pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)
    
    # Create figure for each step
    for i, step in enumerate(steps):
        fig, ax = plt.subplots(figsize=(14, 10))
        visualize_step(step, ax, pos, G)
        plt.tight_layout()
        
        filename = os.path.join(output_folder, f"step_{i+1:02d}_phase{step['phase']}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    print(f"\n✓ Saved {len(steps)} step visualizations to {output_folder}/")


def demonstrate_complement_graph():
    """Demonstrate how complement graph works"""
    print("\n" + "="*80)
    print("COMPLEMENT GRAPH DEMONSTRATION")
    print("="*80)
    
    # Create a simple example
    G_original = nx.Graph()
    G_original.add_edges_from([(1, 2), (1, 3), (2, 4)])
    
    print("\nOriginal Graph G:")
    print(f"  Vertices: {list(G_original.nodes())}")
    print(f"  Edges: {list(G_original.edges())}")
    
    # Create complement
    n = len(G_original.nodes())
    all_possible_edges = []
    nodes = sorted(G_original.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            all_possible_edges.append((nodes[i], nodes[j]))
    
    original_edges_set = set(G_original.edges())
    complement_edges = [e for e in all_possible_edges if e not in original_edges_set]
    
    G_complement = nx.Graph()
    G_complement.add_nodes_from(G_original.nodes())
    G_complement.add_edges_from(complement_edges)
    
    print("\nComplement Graph G':")
    print(f"  Vertices: {list(G_complement.nodes())}")
    print(f"  Edges: {list(G_complement.edges())}")
    
    # Visualize both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    pos = nx.spring_layout(G_original, seed=42)
    
    # Original graph
    nx.draw_networkx_nodes(G_original, pos, node_color=COLOR_VERTEX_NORMAL, 
                          node_size=1000, ax=ax1)
    nx.draw_networkx_edges(G_original, pos, edge_color=COLOR_EDGE_NORMAL, 
                          width=3, ax=ax1)
    nx.draw_networkx_labels(G_original, pos, ax=ax1, font_size=14, font_weight='bold')
    ax1.set_title("Original Graph G", fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Complement graph
    nx.draw_networkx_nodes(G_complement, pos, node_color=COLOR_VERTEX_NORMAL, 
                          node_size=1000, ax=ax2)
    nx.draw_networkx_edges(G_complement, pos, edge_color=COLOR_EDGE_COVERED, 
                          width=3, ax=ax2, style='dashed')
    nx.draw_networkx_labels(G_complement, pos, ax=ax2, font_size=14, font_weight='bold')
    ax2.set_title("Complement Graph G' (dashed edges)", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('complement_graph_example.png', dpi=150, bbox_inches='tight')
    print("\nSaved: complement_graph_example.png")
    
    # Try to show, but save if display not available
    import os
    has_display = os.environ.get('DISPLAY') is not None
    if has_display:
        try:
            plt.show()
        except Exception:
            print("(Display not available, but image saved)")
    else:
        print("(No display available, but image saved)")
    plt.close()


if __name__ == '__main__':
    import sys
    
    print("="*80)
    print("Leaf-Neighbor Greedy Vertex Cover Algorithm - Visualization")
    print("="*80)
    
    # Parse arguments
    filepath = None
    benchmark_name = None
    use_example = False
    mode = 'static'  # default
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--static':
            mode = 'static'
        elif arg == '--complement':
            mode = 'complement'
        elif arg == '--example':
            use_example = True
        elif arg == '--file' and i + 1 < len(sys.argv):
            filepath = sys.argv[i + 1]
            i += 1
        elif arg == '--benchmark' and i + 1 < len(sys.argv):
            benchmark_name = sys.argv[i + 1]
            i += 1
        i += 1
    
    if mode == 'complement':
        # Show complement graph example
        demonstrate_complement_graph()
    elif mode == 'static':
        # Create static images
        if filepath or benchmark_name:
            print(f"\nVisualizing benchmark: {filepath or benchmark_name}")
        elif use_example:
            print("\nUsing simple example graph (18 vertices)")
            print("(Use --file <path> or --benchmark <name> to visualize a benchmark)")
        else:
            print("\nUsing default benchmark: graph50-06")
            print("(Use --example for simple example, --file <path> or --benchmark <name> for other benchmarks)")
        create_static_visualization(filepath=filepath, benchmark_name=benchmark_name, use_example=use_example)
    else:
        # Create animated visualization
        print("\nCreating animated visualization...")
        print("(Use --static to save images, --complement to see complement example)")
        if filepath or benchmark_name:
            print(f"Visualizing benchmark: {filepath or benchmark_name}")
        elif use_example:
            print("Using simple example graph")
        else:
            print("Using default benchmark: graph50-06")
        anim, steps, cover = create_animation(filepath=filepath, benchmark_name=benchmark_name, use_example=use_example)
        
        if anim is not None:
            # Keep animation running
            try:
                input("\nPress Enter to exit...")
            except (EOFError, KeyboardInterrupt):
                print("\n✓ Visualization complete!")
        else:
            print("\n✓ Visualization complete! Check the saved images.")

