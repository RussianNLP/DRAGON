# RAG pipeline reference implementation

import time
from uuid import uuid4

from .helper import log
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def init_retriever(
    dataset,
    embedding_model,
    top_k=5,
    chunk_size=500,
    chunk_overlap=100,
    batch_size=1000,
):
    log("Initializing retriever")
    vector_store = Chroma(embedding_function=embedding_model)
    log("> Creating vector store")
    start_time = time.time()
    documents = [
        Document(id=i, page_content=f"{item['text']}", metadata={"id": item["id"]})
        for i, item in enumerate(dataset["train"])
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    uuids = [str(uuid4()) for _ in range(len(splits))]

    vector_store.reset_collection()
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        batch_ids = uuids[i:i+batch_size]
        vector_store.add_documents(
            documents=batch,
            ids=batch_ids
        )

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

    log(f"> Vector store created for {len(documents)} documents")
    log(f"> Done for {time.time() - start_time:.2f} seconds")

    return retriever


def init_generation(retriever, model, llm_prompt=""):
    log("Initializing generation chain")
    if not llm_prompt:
        llm_prompt = """Answer a question using the provided context. Return answer only.

<context>
{context}
</context>

Question: {input}"""
    prompt = ChatPromptTemplate.from_template(llm_prompt)
    document_chain = create_stuff_documents_chain(model, prompt)
    generation_chain = create_retrieval_chain(retriever, document_chain)
    log("> Done")
    return generation_chain


def prepare_results(data):
    """Prepare your results for evaluation."""
    res = {}
    for item_id, response in data.items():
        res[item_id] = {
            "found_ids": [x.metadata["id"] for x in response["context"]],
            "model_answer": response["answer"],
        }
    return res


def get_results(generation_chain, dataset, skip=None, take=None, write_logs=False, sleep_time=0):
    log("Calculating and preparing results")
    results = {}
    dataset = dataset["train"]
    if skip:
        dataset = dataset.select(range(skip, len(dataset)))
    if take:
        dataset = dataset.select(range(take))
    start_time = time.time()
    for item in tqdm(dataset):
        response = generation_chain.invoke({"input": item["question"]})

        if write_logs:
            print("*" * 80)
            print("id:", item["id"])
            print("Question:", item["question"])
            # print("Answer:", item["answer"])
            print("-----")
            print("Response:", response["answer"])

        results[item["id"]] = response
        if sleep_time > 0:
            time.sleep(sleep_time)
    log(f"Done for {time.time() - start_time:.2f} seconds")
    results = prepare_results(results)
    return results
