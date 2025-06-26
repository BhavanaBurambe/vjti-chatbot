PREVIOUS CODE
import gradio as gr
import os
import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Load FAQs
df = pd.read_csv('vjti_faqs.csv')
df.dropna(inplace=True)
docs = [Document(page_content=row['Answer'], metadata={"question": row['Question']}) for _, row in df.iterrows()]

# Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Embeddings + Vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.from_documents(docs, embeddings)

# Chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

# Chat logic
def ask_chatbot(query):
    result = qa.invoke(query)
    return result['result']

def chatbot_response(user_input, history):
    history = history or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": "⏳ Typing..."})  # temp response
    time.sleep(0.5)
    response = ask_chatbot(user_input)
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

# Gradio App
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 VJTI Helpdesk Chatbot</h1>")

    # chatbot = gr.Chatbot(
    #     type="messages",
    #     height=400,
    #     layout="bubble",
    #     avatar_images=("🧑", "🤖"),
    #     value=[
    #         {"role": "assistant", "content": "Hello! I'm your VJTI Helpdesk Assistant. How can I help you today?"}
    #     ]
    # )

    chatbot = gr.Chatbot(
    type="messages",
    height=400,
    layout="bubble",
    value=[
        {"role": "assistant", "content": "Hello! I'm your VJTI Helpdesk Assistant. How can I help you today?"}
    ]
    )

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask a question...", lines=1)
        send = gr.Button("Send")

    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])

    msg.submit(chatbot_response, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)
    send.click(chatbot_response, inputs=[msg, chatbot], outputs=chatbot).then(lambda: "", None, msg)

# demo.launch()
# for render
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860))) 
