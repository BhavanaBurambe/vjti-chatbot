import gradio as gr
import os
import time
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatDeepInfra
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["DEEPINFRA_API_KEY"] = os.getenv("DEEPINFRA_API_KEY")

# Load CSV
df = pd.read_csv('vjti_faqs.csv')
df.dropna(inplace=True)
docs = [Document(page_content=row['Answer'], metadata={"question": row['Question']}) for _, row in df.iterrows()]

# Embeddings + vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.from_documents(docs, embeddings)

# DeepInfra LLM (Free, hosted)
llm = ChatDeepInfra(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    deepinfra_api_token=os.getenv("DEEPINFRA_API_KEY")
)

# RetrievalQA using LangChain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Main logic
def ask_chatbot(query, history=None):
    history = history or []

    # Check vector similarity
    try:
        matches = vectorstore.similarity_search_with_score(query, k=1)
        if matches and matches[0][1] > 0.7:
            result = qa.invoke(query)
            answer = result.get("result", "").strip()
            if answer:
                return f"📌 *Based on FAQ:*\n\n{answer}"
    except Exception as e:
        print(f"[FAQ error] {e}")

    # If FAQ failed, fallback to LLM
    history_prompt = ""
    for turn in history:
        if turn["role"] == "user":
            history_prompt += f"User: {turn['content']}\n"
        elif turn["role"] == "assistant":
            history_prompt += f"Assistant: {turn['content']}\n"

    full_prompt = (
        "You are the VJTI Helpdesk Assistant. "
        "You can answer both official college FAQs and general queries like location, admission help, etc.\n\n"
        f"{history_prompt}User: {query}\nAssistant:"
    )

    try:
        response = llm.invoke(full_prompt)
        return f"💬 *AI Response:*\n\n{response.content.strip()}"
    except Exception as e:
        print(f"[LLM error] {e}")
        return "⚠️ Sorry, something went wrong while generating a response."

# Gradio chat loop
def chatbot_response(user_input, history):
    history = history or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": "⏳ Typing..."})
    time.sleep(0.5)
    response = ask_chatbot(user_input, history)
    history[-1]["content"] = response
    return history

# Custom CSS
custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
    background: #f8f9fa;
}
.gradio-container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
}
.message {
    word-wrap: break-word;
}
@media only screen and (max-width: 600px) {
    .gradio-container {
        padding: 10px;
    }
}
"""

# Gradio app layout
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 VJTI Helpdesk Chatbot</h1>")

    chatbot = gr.Chatbot(
        type="messages",
        height=400,
        layout="bubble",
        avatar_images=("🧑", "🤖"),
        value=[{"role": "assistant", "content": "Hello! I'm your VJTI Helpdesk Assistant. How can I help you today?"}]
    )

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask a question...", lines=1)
        send = gr.Button("Send")

    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])

    msg.submit(chatbot_response, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)
    send.click(chatbot_response, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)

# For Render deployment
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

# PREVIOUS CODE
# import gradio as gr
# import os
# import time
# import pandas as pd
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
# from langchain.chains import RetrievalQA

# # Load FAQs
# df = pd.read_csv('vjti_faqs.csv')
# df.dropna(inplace=True)
# docs = [Document(page_content=row['Answer'], metadata={"question": row['Question']}) for _, row in df.iterrows()]

# # Google API Key
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# # Embeddings + Vector DB
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# vectorstore = FAISS.from_documents(docs, embeddings)

# # Chat model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

# # Chat logic
# def ask_chatbot(query):
#     result = qa.invoke(query)
#     return result['result']

# def chatbot_response(user_input, history):
#     history = history or []
#     history.append({"role": "user", "content": user_input})
#     history.append({"role": "assistant", "content": "⏳ Typing..."})  # temp response
#     time.sleep(0.5)
#     response = ask_chatbot(user_input)
#     history[-1]["content"] = response
#     return history

# # Custom CSS
# custom_css = """
# body {
#     font-family: 'Segoe UI', sans-serif;
#     background: #f8f9fa;
# }
# .gradio-container {
#     max-width: 900px;
#     margin: auto;
#     padding: 20px;
# }
# .message {
#     word-wrap: break-word;
# }
# @media only screen and (max-width: 600px) {
#     .gradio-container {
#         padding: 10px;
#     }
# }
# """

# # Gradio App
# with gr.Blocks(css=custom_css) as demo:
#     gr.Markdown("<h1 style='text-align: center;'>🤖 VJTI Helpdesk Chatbot</h1>")

#     # chatbot = gr.Chatbot(
#     #     type="messages",
#     #     height=400,
#     #     layout="bubble",
#     #     avatar_images=("🧑", "🤖"),
#     #     value=[
#     #         {"role": "assistant", "content": "Hello! I'm your VJTI Helpdesk Assistant. How can I help you today?"}
#     #     ]
#     # )

#     chatbot = gr.Chatbot(
#     type="messages",
#     height=400,
#     layout="bubble",
#     value=[
#         {"role": "assistant", "content": "Hello! I'm your VJTI Helpdesk Assistant. How can I help you today?"}
#     ]
#     )

#     with gr.Row():
#         msg = gr.Textbox(placeholder="Ask a question...", lines=1)
#         send = gr.Button("Send")

#     with gr.Row():
#         clear = gr.ClearButton([msg, chatbot])

#     msg.submit(chatbot_response, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)
#     send.click(chatbot_response, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)

# # demo.launch()
# # for render
# demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860))) 
