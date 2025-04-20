from langchain_ollama import OllamaEmbeddings

# https://python.langchain.com/docs/integrations/text_embedding/
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings