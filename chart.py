import matplotlib.pyplot as plt
import networkx as nx

def draw_ppo_flowchart():
    G = nx.DiGraph()
    
    # Nodes
    G.add_nodes_from([
        "Initialize Policy Network", "Collect Trajectories", "Compute Advantages",
        "Update Policy", "Clip Objective Function", "Optimize Using Gradient Descent",
        "Policy Update Complete", "Repeat Until Convergence"
    ])
    
    # Edges
    edges = [
        ("Initialize Policy Network", "Collect Trajectories"),
        ("Collect Trajectories", "Compute Advantages"),
        ("Compute Advantages", "Update Policy"),
        ("Update Policy", "Clip Objective Function"),
        ("Clip Objective Function", "Optimize Using Gradient Descent"),
        ("Optimize Using Gradient Descent", "Policy Update Complete"),
        ("Policy Update Complete", "Repeat Until Convergence")
    ]
    G.add_edges_from(edges)
    
    # Draw
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    plt.title("PPO Training Process Flowchart")
    plt.show()

draw_ppo_flowchart()