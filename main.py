from langgraph.graph import StateGraph, END
from agents.query_classifier import classify_and_parse_node, classification_router
from agents.data_retriever import retrieve_node
from agents.visualizer import visualizer_node

def graph_router(state: dict) -> str:
    """Router to determine if visualization is needed after data retrieval"""
    if state.get("graph_needed", False):
        return "visualize"
    else:
        return "__end__"

def build_graph():
    workflow = StateGraph(dict)

    # Add nodes - now including visualizer
    workflow.add_node("classify_and_parse", classify_and_parse_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("visualize", visualizer_node)

    # Set entrypoint
    workflow.set_entry_point("classify_and_parse")

    # Router logic - from classify_and_parse to retrieve or END
    workflow.add_conditional_edges(
        "classify_and_parse",
        classification_router,
        {
            "financial_known": "retrieve",
            "financial_unknown": END,
            "non_financial": END,
            "uncertain": END
        }
    )

    # Router from retrieve to visualize or END based on graph_needed
    workflow.add_conditional_edges(
        "retrieve",
        graph_router,
        {
            "visualize": "visualize",
            "__end__": END
        }
    )

    # Visualizer always goes to END
    workflow.add_edge("visualize", END)

    return workflow.compile()

def run_workflow(user_query: str):
    graph = build_graph()

    initial_state = {
        "query": user_query
    }

    final_state = graph.invoke(initial_state)
    print("\nFinal state:")
    for key, value in final_state.items():
        print(f"{key}: {value}")
    
    # Show graph path if created
    if "graph_path" in final_state:
        print(f"\n📊 Graph created: {final_state['graph_path']}")

if __name__ == "__main__":
    user_query = input("Enter your financial query: ")
    run_workflow(user_query)
