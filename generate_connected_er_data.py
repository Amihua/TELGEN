#!/usr/bin/env python3
"""
TELEGEN - High-Quality Connected ER Graph Data Generator

This script generates training and test datasets for TELEGEN based on the
Erdos-Renyi (ER) random graph model. Its key feature is a strict enforcement
of graph connectivity at the point of generation.

Key Logic:
1. Generate an ER graph with specified parameters (n, p).
2. Immediately check if the graph is connected.
3. If not connected, discard the graph and repeat the generation process.
4. Only once a connected graph is secured, proceed to:
   a. Assign random capacities to edges.
   b. Generate random Source-Destination (SD) pairs and demands.
   c. Find k-shortest paths for each SD pair.
   d. Formulate and solve the corresponding Linear Programming (LP) problem.
5. Save the resulting LP instance if the solver succeeds.
6. Repeat until the desired number of instances for the (n, p) configuration is met.
"""

import os
import gzip
import pickle
import random
import time
import warnings
from pathlib import Path
import argparse

import networkx as nx
import numpy as np
import torch
from scipy.linalg import LinAlgWarning
from scipy.optimize._optimize import OptimizeWarning
from tqdm import tqdm
from itertools import islice

# Assuming 'solver' is a custom module available in the environment
from solver.linprog import linprog

def _parse_int_list(xs):
    out = []
    for x in xs:
        for part in str(x).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def _parse_float_list(xs):
    out = []
    for x in xs:
        for part in str(x).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Generate connected ER TE LP datasets (raw .pkl.gz) with unique SD pairs per instance.")
    ap.add_argument("--root_dir", type=str, default="./data/raw", help="Output directory for raw instances")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--k_paths", type=int, default=4, help="k-shortest paths per SD")
    ap.add_argument("--sd_pairs", type=int, default=10, help="SD pairs per instance (unique within instance)")
    ap.add_argument("--capacity_low", type=float, default=1000.0)
    ap.add_argument("--capacity_high", type=float, default=5000.0)
    ap.add_argument("--demand_low", type=float, default=1000.0)
    ap.add_argument("--demand_high", type=float, default=5000.0)
    ap.add_argument("--bounds_low", type=float, default=0.0)
    ap.add_argument("--bounds_high", type=float, default=1.0)

    ap.add_argument("--train_nodes", nargs="*", default=list(range(20, 101, 10)))
    ap.add_argument("--train_probs", nargs="*", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--train_instances", type=int, default=200)

    ap.add_argument("--test_nodes", nargs="*", default=[200, 500, 1000, 2000])
    ap.add_argument("--test_probs", nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--test_instances", type=int, default=300)

    ap.add_argument("--only", choices=["train", "test", "all"], default="all")
    ap.add_argument("--smoke", action="store_true", help="Run a tiny smoke generation (overrides train/test specs)")
    return ap.parse_args()


# --- Core Functions (adapted from Notebook) ---

def generate_connected_graph(n, p, seed):
    """
    Generates a NetworkX graph that is guaranteed to be connected.
    It repeatedly generates ER graphs until a connected one is found.
    """
    attempt = 0
    while True:
        # Use a new seed for each attempt to ensure variety
        er_graph = nx.fast_gnp_random_graph(n, p, seed=seed + attempt, directed=False)
        if nx.is_connected(er_graph):
            return er_graph
        attempt += 1
        if attempt > 1000: # Safety break to prevent infinite loops
            raise RuntimeError(f"Failed to generate a connected graph for n={n}, p={p} after 1000 attempts.")

def assign_random_capacities(graph, capacity_range):
    """Assigns random capacities to each edge in the graph."""
    for u, v in graph.edges():
        capacity = random.uniform(*capacity_range)
        graph[u][v]['capacity'] = capacity
    return graph

def k_shortest_paths(G, source, target, k, weight=None):
    """Calculates k-shortest simple paths."""
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def generate_lp_problem(G, std_pairs, k_paths_dict, k):
    """
    Generates the LP matrices (A, b, c) based on the graph and demands.
    This function is adapted from the `generate_reallocation` logic in the notebook.
    """
    num_sd_pairs = len(std_pairs)
    edges_list = list(G.edges())
    num_edges = len(edges_list)
    num_vars = num_sd_pairs * k

    # Constraint 1: Sum of flows for each demand is 1
    A1 = np.zeros((num_sd_pairs, num_vars))
    for i in range(num_sd_pairs):
        A1[i, k*i : k*i+k] = 1
    b1 = np.ones(num_sd_pairs)

    # Constraint 2: Capacity constraints
    A2 = np.zeros((num_edges, num_vars))
    edge_to_idx = {edge: i for i, edge in enumerate(edges_list)}

    for i, (st, demand) in enumerate(std_pairs):
        paths = k_paths_dict[tuple(st)]
        for j, path in enumerate(paths):
            for u, v in zip(path[:-1], path[1:]):
                edge = (u, v) if (u, v) in edge_to_idx else (v, u)
                if edge in edge_to_idx:
                    A2[edge_to_idx[edge], k*i+j] = demand
    
    b2_dict = nx.get_edge_attributes(G, 'capacity')
    b2 = np.array([b2_dict[edge] for edge in edges_list])

    # Remove zero rows which can occur if an edge is not used by any k-shortest path
    non_zero_rows = np.any(A2, axis=1)
    A2 = A2[non_zero_rows]
    b2 = b2[non_zero_rows]
    
    # Normalize capacity constraints
    b2_inv = 1.0 / b2
    A2 = A2 * b2_inv[:, np.newaxis]
    b2.fill(1.0)

    # Objective function: Maximize total routed demand
    c = -1 * np.concatenate([np.ones(k) * demand for _, demand in std_pairs])
    
    # Combine matrices
    A = np.vstack([A1, A2])
    b = np.hstack([b1, b2])

    return A, b, c

def generate_instances_for_config(n, p, num_instances, prefix, *, root_dir: Path, seed: int, k_paths: int, sd_pairs: int, cap_range, demand_range, bounds):
    """
    Main generation loop for a single (n, p) configuration.
    """
    print(f"\n--- Generating {prefix} set for n={n}, p={p} ---")
    
    filename = root_dir / f"instance_ER_{prefix}_p{p}_n{n}.pkl.gz"
    
    lp_instances = []
    
    pbar = tqdm(total=num_instances, desc=f"Connected Instances for n={n}, p={p}")
    
    while len(lp_instances) < num_instances:
        # 1. Generate a CONNECTED graph
        graph_seed = seed + n + int(p * 100) + len(lp_instances)
        base_graph = generate_connected_graph(n, p, seed=graph_seed)
        graph = assign_random_capacities(base_graph, cap_range)

        # 2. Generate SD pairs and find k-shortest paths
        std_pairs = []
        k_paths_dict = {}
        nodes = list(graph.nodes())
        attempts = 0
        while len(std_pairs) < sd_pairs and attempts < 100 * sd_pairs:
            st = tuple(random.sample(nodes, 2))
            if st in k_paths_dict:
                # Ensure SD pairs are unique; count attempts to avoid potential infinite loops
                attempts += 1
                continue
            
            paths = k_shortest_paths(graph, st[0], st[1], k=k_paths)
            
            if len(paths) == k_paths:
                demand = random.uniform(*demand_range)
                std_pairs.append((st, demand))
                k_paths_dict[st] = paths
            attempts += 1
        
        if len(std_pairs) < sd_pairs:
            print(f"\n[Warning] Could not find {k_paths} paths for {sd_pairs} SD pairs in a connected graph. Skipping this graph.")
            continue

        # 3. Formulate and solve the LP
        A, b, c = generate_lp_problem(graph, std_pairs, k_paths_dict, k_paths)
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='interior-point')
            if res.success and not np.isnan(res.fun):
                instance = (torch.from_numpy(A).to(torch.float),
                            torch.from_numpy(b).to(torch.float),
                            torch.from_numpy(c).to(torch.float))
                lp_instances.append(instance)
                pbar.update(1)
        except (LinAlgWarning, OptimizeWarning, ValueError) as e:
            # Catch solver errors and continue
            print(f"\n[Solver Warning] for n={n}, p={p}: {e}. Skipping instance.")
            continue

    pbar.close()

    # 4. Save the collected instances to a file
    print(f"Saving {len(lp_instances)} instances to {filename}...")
    with gzip.open(filename, "wb") as file:
        pickle.dump(lp_instances, file)
    print("Save complete.")

def main():
    """
    Main function to run the entire data generation pipeline.
    """
    args = parse_args()

    train_nodes = _parse_int_list(args.train_nodes)
    train_probs = _parse_float_list(args.train_probs)
    test_nodes = _parse_int_list(args.test_nodes)
    test_probs = _parse_float_list(args.test_probs)

    if args.smoke:
        train_nodes = [20]
        train_probs = [0.5]
        test_nodes = [200]
        test_probs = [0.5]
        args.train_instances = 5
        args.test_instances = 5

    root_dir = Path(args.root_dir)

    print("🎯 Starting High-Quality ER Graph Data Generation")
    print("=" * 60)
    
    # Set global random seeds
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    # Create the output directory if it doesn't exist
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Generate Training Set ---
    cap_range = (float(args.capacity_low), float(args.capacity_high))
    demand_range = (float(args.demand_low), float(args.demand_high))
    bounds = (float(args.bounds_low), float(args.bounds_high))

    if args.only in ("train", "all"):
        print("\n>>> GENERATING TRAINING SET <<<")
        for p in train_probs:
            for n in train_nodes:
                generate_instances_for_config(
                    n,
                    p,
                    int(args.train_instances),
                    "train",
                    root_dir=root_dir,
                    seed=int(args.seed),
                    k_paths=int(args.k_paths),
                    sd_pairs=int(args.sd_pairs),
                    cap_range=cap_range,
                    demand_range=demand_range,
                    bounds=bounds,
                )
            
    # --- Generate Test Set ---
    if args.only in ("test", "all"):
        print("\n>>> GENERATING TEST SET <<<")
        for p in test_probs:
            for n in test_nodes:
                generate_instances_for_config(
                    n,
                    p,
                    int(args.test_instances),
                    "test",
                    root_dir=root_dir,
                    seed=int(args.seed),
                    k_paths=int(args.k_paths),
                    sd_pairs=int(args.sd_pairs),
                    cap_range=cap_range,
                    demand_range=demand_range,
                    bounds=bounds,
                )
            
    print("\n🎉 All data generation tasks completed successfully!")

if __name__ == "__main__":
    # Suppress warnings from the solver for cleaner output
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    main()

