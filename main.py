#!/usr/bin/env python3
"""
Main script to run vertex cover algorithm on all benchmarks and update result.txt
This script does everything: processes benchmarks, computes results, and updates result.txt
"""

import os
import sys
import time
from collections import defaultdict
import glob

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Will only process .txt files.")


# ============================================================================
# ALGORITHM IMPLEMENTATION
# ============================================================================

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


def build_graph(edges):
    """Build adjacency list representation of the graph"""
    graph = defaultdict(set)
    degree = defaultdict(int)
    
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        degree[u] += 1
        degree[v] += 1
    
    return graph, degree


def compute_complement_graph(edges, vertices):
    """
    Compute the complement graph G' of graph G.
    In the complement graph, an edge exists between two vertices
    if and only if it does NOT exist in the original graph.
    """
    # Create set of original edges for fast lookup
    original_edges = set()
    for u, v in edges:
        if u < v:
            original_edges.add((u, v))
        else:
            original_edges.add((v, u))
    
    # Generate all possible edges in complete graph
    vertices_list = sorted(vertices)
    n = len(vertices_list)
    complement_edges = []
    
    # For each pair of vertices, add edge if not in original graph
    for i in range(n):
        for j in range(i + 1, n):
            u, v = vertices_list[i], vertices_list[j]
            edge_tuple = (u, v) if u < v else (v, u)
            if edge_tuple not in original_edges:
                complement_edges.append((u, v))
    
    return complement_edges


def greedy_vertex_cover(graph, degree):
    """
    Greedy Vertex Cover Algorithm:
    1. While there are leaves (degree 1), remove their neighbors
    2. When no leaves remain, remove highest-degree nodes
    """
    # Create copies to avoid modifying original
    g = {v: set(neighbors) for v, neighbors in graph.items()}
    deg = degree.copy()
    
    vertex_cover = set()
    remaining_vertices = set(g.keys())
    
    # Phase 1: Remove neighbors of leaves
    while True:
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
    
    # Phase 2: Remove highest-degree nodes until no edges remain
    while remaining_vertices:
        # Find all remaining edges
        remaining_edges = []
        for v in remaining_vertices:
            for u in g[v]:
                if u in remaining_vertices and u > v:  # Avoid duplicates
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
    
    return len(vertex_cover)


def is_red_benchmark(benchmark_name):
    """Check if a benchmark requires complement graph conversion"""
    red_benchmarks = {
        'gen200_p0.9_44',
        'phat300_1_c',
        'phat300_2_c',
        'mann_a27',
        'sanr400_0.5',
        'sanr400_0.7'
    }
    return benchmark_name in red_benchmarks


def process_benchmark(filepath, needs_complement=False):
    """Process a single benchmark file and return results"""
    try:
        start_time = time.time()
        
        if filepath.endswith('.txt'):
            edges, vertices = read_graph_from_txt(filepath)
        elif filepath.endswith('.xlsx'):
            edges, vertices = read_graph_from_xlsx(filepath)
        else:
            return None
        
        if not edges:
            return None
        
        n = len(vertices)
        
        # For red benchmarks: compute on complement graph
        # For regular benchmarks: compute on original graph, then also compute on complement for ω(G)
        if needs_complement:
            # Red benchmark: compute τ(G') where G' is complement
            complement_edges = compute_complement_graph(edges, vertices)
            graph, degree = build_graph(complement_edges)
            tau_complement = greedy_vertex_cover(graph, degree)
            
            # For red benchmarks:
            # τ(G') = tau_complement (vertex cover of complement)
            # α(G') = n - τ(G') (independent set of complement)
            # ω(G) = α(G') = n - τ(G') (clique of original = independent set of complement)
            # τ(G) = n - α(G) = n - ω(G') where G' is complement... actually, we need τ(G) of original
            # For red benchmarks, we compute τ(G') directly, so:
            # ω(G) = n - τ(G')
            # To get τ(G) of original, we'd need to compute it separately or use relationship
            
            # Actually, for red benchmarks, the user wants:
            # τ(G) = we compute on complement, so this is τ(G')
            # But wait, let me check the user's requirement again
            
            # Based on the reducibility: if we computed τ(G'), then:
            # ω(G) = α(G') = n - τ(G')
            # For τ(G) of the original graph, we need to compute it on the original
            original_graph, original_degree = build_graph(edges)
            tau_original = greedy_vertex_cover(original_graph, original_degree)
            
            execution_time = time.time() - start_time
            
            return {
                'file': os.path.basename(filepath),
                'total_vertices': n,
                'tau_g': tau_original,  # τ(G) of original graph
                'alpha_g': n - tau_original,  # α(G) = n - τ(G)
                'omega_g': n - tau_complement,  # ω(G) = α(G') = n - τ(G')
                'execution_time': execution_time,
                'is_red': True
            }
        else:
            # Regular benchmark: compute τ(G) on original
            graph, degree = build_graph(edges)
            tau_g = greedy_vertex_cover(graph, degree)
            
            # Also compute on complement to get ω(G)
            complement_edges = compute_complement_graph(edges, vertices)
            complement_graph, complement_degree = build_graph(complement_edges)
            tau_complement = greedy_vertex_cover(complement_graph, complement_degree)
            
            execution_time = time.time() - start_time
            
            return {
                'file': os.path.basename(filepath),
                'total_vertices': n,
                'tau_g': tau_g,  # τ(G)
                'alpha_g': n - tau_g,  # α(G) = n - τ(G)
                'omega_g': n - tau_complement,  # ω(G) = α(G') = n - τ(G')
                'execution_time': execution_time,
                'is_red': False
            }
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return None


# ============================================================================
# BENCHMARK NAME MAPPING
# ============================================================================

def normalize_benchmark_name(name):
    """Normalize benchmark name for matching"""
    name = name.lower().replace('_', '-').replace(' ', '-')
    name = name.replace('.clq-compliment', '').replace('.clq.b', '')
    name = name.replace('.xlsx', '').replace('.txt', '')
    return name


def get_benchmark_mapping():
    """Map result.txt benchmark names to actual file patterns"""
    mapping = {
        'jhonson8_2_4': ['johnson8-2-4', 'johnson8_2_4'],
        'graph50_6': ['graph50-06', 'graph50_6'],
        'graph50_10': ['graph50-10', 'graph50_10'],
        'Hamming6_2': ['hamming6-2', 'hamming6_2'],
        'Hamming6_4': ['hamming6-4', 'hamming6_4', 'hamming8-4'],
        'jhonson8_4_4': ['johnson8-4-4', 'johnson8_4_4'],
        'graph100_1': ['graph100-01', 'graph100_1'],
        'graph100_10': ['graph100-10', 'graph100_10'],
        'jhonson16_2_4': ['johnson16-2-4', 'johnson16_2_4'],
        'c125': ['c125.9', 'c125', 'C125.9'],
        'keller4': ['keller4', 'keller4_c'],
        'graph200_5': ['graph200-05', 'graph200_5'],
        'broc200_2': ['broc 200-2', 'broc200-2', 'broc200_2'],
        'broc200_4': ['broc 200-4', 'broc200-4', 'broc200_4'],
        'cfat200_1': ['c-fat200-1', 'cfat200-1', 'cfat200_1'],
        'cfat200_2': ['c-fat200-2', 'cfat200-2', 'cfat200_2'],
        'cfat200_5': ['c-fat200-5', 'cfat200-5', 'cfat200_5'],
        'sanr200-0.7': ['sanr200_0.7', 'sanr200-0.7'],
        'gen200_p0.9_44': ['gen200_p0.9_44', 'gen200_p0.9_44.b'],
        'C250': ['c250.9', 'C250.9', 'c250'],
        'Hamming8_2': ['hamming8-2', 'hamming8_2'],
        'Hamming8_4': ['hamming8-4', 'hamming8_4'],
        'phat300_1': ['phat300-1', 'phat300_1'],
        'phat300_1_c': ['phat300-2', 'phat300_2', 'phat300-3'],
        'phat300_2_c': ['phat300-3', 'phat300_3'],
        'mann_a27': ['mann_a27', 'MANN_a27'],
        'sanr400_0.5': ['sanr400_0.5', 'sanr400-0.5'],
        'sanr400_0.7': ['sanr400_0.7', 'sanr400-0.7'],
        'jhonson32_2_4_c': ['johnson32-2-4', 'johnson32_2_4'],
        'graph500_1': ['graph500-01', 'graph500_1'],
        'graph500_2': ['graph500-02', 'graph500_2'],
        'graph500_5': ['graph500-05', 'graph500_5'],
        'cfat500_1': ['c-fat500-1', 'cfat500-1', 'cfat500_1'],
    }
    return mapping


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def read_result_txt():
    """Read the current result.txt file and extract benchmark information"""
    result_file = os.path.join(os.path.dirname(__file__), 'result.txt')
    benchmarks = []
    
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found. Creating new one.")
        return benchmarks
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Skip header, separator, and empty lines
            if '|' in line and not line.strip().startswith('|--') and 'Benchmark' not in line and '|V|' not in line:
                parts = [p.strip() for p in line.split('|')]
                # Check if this is a valid benchmark row (has at least name and vertices)
                # New format: Benchmarks | |V| | Optimal MVC | τ(G) | α(G) | ω(G) | t
                if len(parts) >= 3 and parts[1] and parts[1] != 'Benchmark' and parts[1] != '|V|':
                    benchmark_name = parts[1].strip()
                    # Skip if empty name
                    if not benchmark_name:
                        continue
                    
                    # New format: parts[2] = |V|, parts[3] = Optimal MVC
                    vertices = parts[2] if len(parts) > 2 else ''
                    opt = parts[3] if len(parts) > 3 else ''
                    
                    try:
                        vertices = int(vertices) if vertices else 0
                        opt = int(opt) if opt and opt != 'N/A' else None
                    except:
                        vertices = 0
                        opt = None
                    
                    benchmarks.append({
                        'name': benchmark_name,
                        'vertices': vertices,
                        'opt': opt
                    })
    
    return benchmarks


def find_matching_file(benchmark_name, all_files):
    """Find the actual benchmark file that matches the benchmark name"""
    mapping = get_benchmark_mapping()
    normalized_target = normalize_benchmark_name(benchmark_name)
    
    # Check direct mapping
    if benchmark_name in mapping:
        patterns = mapping[benchmark_name]
        for pattern in patterns:
            pattern_norm = normalize_benchmark_name(pattern)
            for filepath in all_files:
                filename = os.path.basename(filepath)
                filename_norm = normalize_benchmark_name(filename)
                if pattern_norm in filename_norm or filename_norm in pattern_norm:
                    return filepath
    
    # Try fuzzy matching
    for filepath in all_files:
        filename = os.path.basename(filepath)
        filename_norm = normalize_benchmark_name(filename)
        
        # Check if normalized names match
        if normalized_target in filename_norm or filename_norm in normalized_target:
            return filepath
        
        # Check partial matches
        if len(normalized_target) > 5 and normalized_target[:10] in filename_norm:
            return filepath
    
    return None


def update_result_txt(benchmark_results, benchmarks):
    """Update result.txt with new format: Benchmarks | |V| | Optimal MVC | τ(G) | α(G) | ω(G) | t"""
    result_file = os.path.join(os.path.dirname(__file__), 'result.txt')
    
    # Create result map
    result_map = {}
    for br in benchmark_results:
        result_map[br['benchmark_name']] = br
    
    # Create benchmark map for optimal values
    benchmark_map = {}
    for b in benchmarks:
        benchmark_map[b['name']] = b
    
    # Write new format
    output_lines = []
    # Header: Benchmarks | |V| | Optimal MVC | τ(G) | α(G) | ω(G) | t
    output_lines.append("| Benchmarks           | |V| | Optimal MVC | τ(G) | α(G) | ω(G) | t          |\n")
    output_lines.append("|---------------------|-----|--------------|------|------|------|------------|\n")
    
    # Process each benchmark
    for benchmark in benchmarks:
        benchmark_name = benchmark['name']
        vertices = benchmark['vertices']
        opt_mvc = benchmark['opt']
        
        result = result_map.get(benchmark_name, {})
        tau_g = result.get('tau_g', None)
        alpha_g = result.get('alpha_g', None)
        omega_g = result.get('omega_g', None)
        exec_time = result.get('execution_time', None)
        
        # Format values
        vertices_str = str(vertices) if vertices else ''
        opt_str = str(opt_mvc) if opt_mvc is not None else ''
        tau_str = str(tau_g) if tau_g is not None else ''
        alpha_str = str(alpha_g) if alpha_g is not None else ''
        omega_str = str(omega_g) if omega_g is not None else ''
        time_str = f"{exec_time:.4f}s" if exec_time is not None else ''
        
        # Check if this is a red benchmark
        is_red = is_red_benchmark(benchmark_name)
        # Add (red) marker for red benchmarks
        benchmark_display = f"(red) {benchmark_name}" if is_red else benchmark_name
        
        # Format line
        output_lines.append(f"| {benchmark_display:<21} | {vertices_str:<3} | {opt_str:<12} | {tau_str:<4} | {alpha_str:<4} | {omega_str:<4} | {time_str:<10} |\n")
    
    # Write updated file
    with open(result_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"\n✓ Updated {result_file} with computed values")


def main():
    """Main function to process all benchmarks and update result.txt"""
    # Record start time
    start_time = time.time()
    
    print("="*80)
    print("Vertex Cover Algorithm - Benchmark Processing")
    print("="*80)
    
    # Get benchmark directory
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'Benchamarks-20251129', 'Benchamarks')
    
    if not os.path.exists(benchmark_dir):
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return
    
    # Get all benchmark files
    txt_files = glob.glob(os.path.join(benchmark_dir, '*.txt'))
    xlsx_files = glob.glob(os.path.join(benchmark_dir, '*.xlsx'))
    all_files = sorted(txt_files + xlsx_files)
    
    print(f"\nFound {len(all_files)} benchmark files")
    
    # Read result.txt to get benchmark list
    benchmarks = read_result_txt()
    print(f"Found {len(benchmarks)} benchmarks in result.txt")
    
    if not benchmarks:
        print("No benchmarks found in result.txt. Processing all files...")
        # Process all files and create new result.txt
        results = []
        for filepath in all_files:
            print(f"Processing: {os.path.basename(filepath)}...", end=' ', flush=True)
            result = process_benchmark(filepath, needs_complement=False)
            if result:
                results.append(result)
                tau_g = result.get('tau_g', 'N/A')
                print(f"τ(G): {tau_g}")
        return
    
    # Process benchmarks
    print("\nProcessing benchmarks...")
    benchmark_results = []
    
    for i, benchmark in enumerate(benchmarks, 1):
        benchmark_name = benchmark['name']
        needs_complement = is_red_benchmark(benchmark_name)
        complement_str = " [COMPLEMENT]" if needs_complement else ""
        print(f"[{i}/{len(benchmarks)}] {benchmark_name}{complement_str}...", end=' ', flush=True)
        
        # Find matching file
        matching_file = find_matching_file(benchmark_name, all_files)
        
        if not matching_file:
            print("FILE NOT FOUND (missing from Benchamarks/ folder)")
            # Still add to results with None to indicate missing file
            benchmark_results.append({
                'benchmark_name': benchmark_name,
                'tau_g': None,
                'alpha_g': None,
                'omega_g': None,
                'execution_time': None
            })
            continue
        
        # Process the file
        result = process_benchmark(matching_file, needs_complement=needs_complement)
        
        if result:
            tau_g = result.get('tau_g', None)
            alpha_g = result.get('alpha_g', None)
            omega_g = result.get('omega_g', None)
            exec_time = result.get('execution_time', None)
            opt = benchmark['opt']
            
            benchmark_results.append({
                'benchmark_name': benchmark_name,
                'tau_g': tau_g,
                'alpha_g': alpha_g,
                'omega_g': omega_g,
                'execution_time': exec_time
            })
            
            time_str = f", Time: {exec_time:.4f}s" if exec_time else ""
            print(f"τ(G): {tau_g}, α(G): {alpha_g}, ω(G): {omega_g}{time_str}")
        else:
            print("FAILED")
            benchmark_results.append({
                'benchmark_name': benchmark_name,
                'tau_g': None,
                'alpha_g': None,
                'omega_g': None,
                'execution_time': None
            })
    
    # Update result.txt
    if benchmark_results:
        update_result_txt(benchmark_results, benchmarks)
        print(f"\n✓ Successfully processed {len(benchmark_results)} benchmarks")
    else:
        print("\n✗ No results to update")
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format time nicely
    if execution_time < 60:
        time_str = f"{execution_time:.2f} seconds"
    elif execution_time < 3600:
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        time_str = f"{minutes} minute(s) {seconds:.2f} seconds"
    else:
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        time_str = f"{hours} hour(s) {minutes} minute(s) {seconds:.2f} seconds"
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Total execution time: {time_str}")
    print("="*80)


if __name__ == '__main__':
    main()

