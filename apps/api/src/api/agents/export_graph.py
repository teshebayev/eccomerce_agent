from api.agents.graph import graph
from api.agents.export_langgraph_png import save_langgraph_visualization

if __name__ == "__main__":
    path = save_langgraph_visualization(graph, "/tmp/langgraph_visualization.png")
    print(f"Saved to: {path}")