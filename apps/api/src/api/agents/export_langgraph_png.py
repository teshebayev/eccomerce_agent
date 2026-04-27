from pathlib import Path


def save_langgraph_visualization(graph, output_path=None):
    if output_path is None:
        output_path = Path("/tmp/langgraph_visualization.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_bytes = graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_bytes)

    return output_path


if __name__ == "__main__":
    # Example usage:
    # from api.agents.graph import graph
    # path = save_langgraph_visualization(graph)
    # print(f"Saved to: {path}")
    pass
