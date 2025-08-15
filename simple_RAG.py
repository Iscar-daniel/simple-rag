import os

from haystack.components.agents import Agent
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch
from haystack.utils import Secret
from haystack.components.generators.utils import print_streaming_chunk
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder


# search_tool = ComponentTool(component=SerperDevWebSearch())

# Write documents to InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Melisa and I live in Waingapu."),
    Document(content="My name is Joko and I live in Solo."),
    Document(content="My name is Giorgio and I live in Santorini.")
])

# Build a RAG pipeline
prompt_template = """
        You are a helpful assistant
        "Given these documents, answer the question in indonesia language.\n"
        "Documents:\n{% for doc in documents %}{{ doc.content }}{% endfor %}\n"
        "Question: {{question}}\n"
        "Answer:"
        """
# Define required variables explicitly
prompt_builder = PromptBuilder(template=prompt_template , required_variables={"question", "documents"})

retriever = InMemoryBM25Retriever(document_store=document_store)
chat_generator=OpenAIGenerator(model="lgai/exaone-3-5-32b-instruct", 
                                       api_key=Secret.from_env_var("TOGETHER_AI_API_KEY"),
                                       api_base_url="https://api.together.xyz/v1",
                                    #    streaming_callback=print_streaming_chunk
                                       )
answer_builder = AnswerBuilder(pattern="(.*) tinggal di ")


rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder",answer_builder)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies","answer_builder.replies")
rag_pipeline.connect("retriever","answer_builder.documents")
rag_pipeline.connect("llm.meta","answer_builder.meta")
rag_pipeline.draw(path="image2.png")
# rag_pipeline.connect("prompt", "chat_generator")

#test asking question
question= "who lives in waingapu?"
results= rag_pipeline.run({
     "retriever": {"query": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
})

print(results['answer_builder']['answers'][0].meta['all_messages']) 