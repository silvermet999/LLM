from langchain_ollama import OllamaEmbeddings

# https://python.langchain.com/docs/integrations/text_embedding/
# https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama2:latest")
    return embeddings