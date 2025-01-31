# RAG
RAG - retrieval augmented generation via language models

via LangChain, HuggingFace, and FAISS

using local LLM, "mistralai/Mistral-7B-Instruct-v0.1" (local)

The following is a snippet of the jupyer code file.

Loading the local LLM to GPU RAM:
```
modelID = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                               )
model101 = AutoModelForCausalLM.from_pretrained(modelID, quantization_config=bnb_config, device_map="auto")
tokenizer101 = AutoTokenizer.from_pretrained(modelID)
```

HuggingFace text-generation pipeline:
```
# create the LLM pipeline 
text_generation_pipeline = transformers.pipeline(
    model=model101,
    tokenizer=tokenizer101,
    task="text-generation",
    eos_token_id=tokenizer101.eos_token_id,
    pad_token_id=tokenizer101.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
)
#
HFP_mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
```

Create numeric vectors for vectorDB, via embedding model from HuggingFace:
```
# This small sentence-transformer model is able to convert text strings into a vector representation; we will use it for our vector database.
from langchain.embeddings import HuggingFaceEmbeddings
embeddings101 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={"device": "cuda"},
)
```

Create vectorDB via LangChain and FAISS:
```
# create a vector database and a VectorStoreRetriever object 
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
vector_db101 = FAISS.from_texts(db_docs101, embeddings101)
retriever101 = VectorStoreRetriever(vectorstore=vector_db101)
```

Create the RAG for querstion&answer:
```
# create a RetrievalQA object, which is specially designed for question-answering: 
template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
{context}
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Chat history: {history}
Question: {question}
Write your answers short. Helpful Answer:"""
#
prompt303 = PromptTemplate(
        template=template, input_variables=["history", "context", "question"]
    )
#
RAQA303 = RetrievalQA.from_chain_type(
        llm=HFP_mistral_llm,
        chain_type="stuff",
        retriever=retriever101,
        chain_type_kwargs={
            #"verbose": False,
            "verbose": True,
            "prompt": prompt303,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
    )
```

ASk Q and get A:
```
iquestion = "What is the tire diameter of Airbus A380 in centimeters?"
#
RAGresponse303 = RAQA303.run(iquestion)
#
print()
print('iquestion {{{' ,iquestion, "}}}")
print('RAGresponse303 {{{' ,RAGresponse303, "}}}")
```

```
Human: What is the weight of Airbus A380 in kilograms?
AI:  The weight of Airbus A380 is approximately 200,000 kg (440,000 lb).
33 Wed Dec 20 18:52:05 2023
```

