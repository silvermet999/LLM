from langchain.document_loaders.pdf import PyPDFDirectoryLoader

file_path = "LawsOfChess.pdf"
def load_documents():
    document_loader = PyPDFDirectoryLoader(file_path)
    return document_loader.load()

