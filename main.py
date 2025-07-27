from langgraph.graph import StateGraph, END
from agents.query_classifier import classify_and_parse_node, classification_router
from agents.data_retriever import retrieve_node
from agents.aggregator import aggregator_node
from agents.visualizer import visualizer_node

def aggregation_router(state: dict) -> str:
    """Router to determine if aggregation is needed after data retrieval"""
    aggregations = state.get("aggregations", [])
    if aggregations and any(agg.get("aggregation") for agg in aggregations):
        return "aggregate"
    elif state.get("graph_needed", False):
        return "visualize"
    else:
        return "__end__"

def graph_router(state: dict) -> str:
    """Router to determine if visualization is needed after aggregation"""
    if state.get("graph_needed", False):
        return "visualize"
    else:
        return "__end__"

def build_graph():
    workflow = StateGraph(dict)

    # Add nodes - now including aggregator
    workflow.add_node("classify_and_parse", classify_and_parse_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("aggregate", aggregator_node)
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

    # Router from retrieve to aggregate, visualize, or END
    workflow.add_conditional_edges(
        "retrieve",
        aggregation_router,
        {
            "aggregate": "aggregate",
            "visualize": "visualize",
            "__end__": END
        }
    )

    # Router from aggregate to visualize or END based on graph_needed
    workflow.add_conditional_edges(
        "aggregate",
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
    
    # # Show aggregated results if available
    # if "aggregated_data" in final_state:
    #     print(f"\n📊 Aggregated results available")
    #     if not final_state["aggregated_data"].get("data").empty:
    #         print(final_state["aggregated_data"]["data"])
    
    # Show graph path if created
    if "graph_path" in final_state:
        print(f"\n📊 Graph created: {final_state['graph_path']}")

if __name__ == "__main__":
    user_query = input("Enter your financial query: ")
    run_workflow(user_query)
