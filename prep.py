from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# https://python.langchain.com/docs/how_to/#document-loaders
file_path = "data"
def load_documents():
    document_loader = PyPDFDirectoryLoader(file_path)
    return document_loader.load()
documents = load_documents()
# print(documents[0])


# https://python.langchain.com/docs/how_to/recursive_text_splitter/
def text_splitter(document : list[documents]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # max num of char per chunks
        chunk_overlap=100, # chunk overlapping for context: 10–20% of chunk_size
        length_function=len,
        is_separator_regex=True
    )
    return text_splitter.split_documents(document)

text = text_splitter(documents)


# https://python.langchain.com/docs/integrations/vectorstores/chroma/
def add_to_chroma(embeddings):
    vector_store = Chroma(
        # collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    return vector_store

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
