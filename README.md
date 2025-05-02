```bash
python prep.py
```

prep.py contains 3 functions:
load_documents() load the pdf file
text_splitter(documents) pass the loaded document to split it into chunks
calculate_chunk_ids(chunks) assign IDs to chunks
add_to_chroma(embedding, chunks) pass the embedding and the chunks into the Chroma database.
The embedding model is in embedding.py . We used Llama2:latest from Ollama.

```bash
python query.py
```