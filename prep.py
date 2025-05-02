from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embedding import get_embedding_function


# https://python.langchain.com/docs/how_to/#document-loaders
file_path = "data"
chroma_path = "chroma_langchain_db"

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

chunks = text_splitter(documents)


# https://python.langchain.com/docs/integrations/vectorstores/chroma/
def add_to_chroma(embeddings, chunks: list[documents]):
    vector_store = Chroma(
        # collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=chroma_path,  # Where to save data locally, remove if not necessary
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = vector_store.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


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

embedding = get_embedding_function()
add_to_chroma(embedding, chunks)