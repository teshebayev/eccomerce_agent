
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Document, FusionQuery, Prefetch


client = OpenAI(
    base_url="http://ollama:11434/v1",
    api_key="ollama",
)

# @traceable(
#     name="embedding_query",
#     run_type="embedding",
#     metadata={
#         "ls_provider": "ollama",
#         "ls_model_name": "nomic-embed-text",
#     },
# )
def get_embedding(text, model="nomic-embed-text"):
    response = client.embeddings.create(
        input=text,
        model=model,
    )

    # run = get_current_run_tree()
    # if run and getattr(response, "usage", None):
    #     run.set(
    #         usage_metadata={
    #             "input_tokens": response.usage.prompt_tokens,
    #             # для embeddings output_tokens обычно нет
    #             "total_tokens": response.usage.total_tokens,
    #         }
    #     )

    return response.data[0].embedding


# @traceable(
#     name="retrieve_data",
#     run_type="retriever",
# )

def retrieve_data(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    search_result = qdrant_client.query_points(
        collection_name="Amazon_items_collection1_hybrid_search",
        prefetch = [
            Prefetch(
             query=query_embedding,
             using="nomic-embed-text",
             limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm-25",
                limit=20
                )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )

    retrieved_context_ids = []
    retrieved_contexts = []
    similarity_scores = []
    retrieved_context_ratings = []

    for point in search_result.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_contexts.append(point.payload["description"])
        retrieved_context_ratings.append(point.payload["average_rating"])
        similarity_scores.append(point.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_contexts": retrieved_contexts,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }


def process_context(context):
    processed_context = ""
    for id_, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_contexts"],
        context["retrieved_context_ratings"]
    ):
        processed_context += f"-ID {id_}: rating: {rating}, description: {chunk}\n"
    return processed_context


def get_formatted_context(query, top_k=5):
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query (str): The user query to get top k context for.
        top_k (int, optional): The number of top context to retrieve. Defaults to 5.
    
    Returns:
        str: A formatted string containing the top k context with IDs and average ratings, each representing an inventory item.

    """
    context = retrieve_data(query,top_k)
    formatted_context = process_context(context)
    return formatted_context