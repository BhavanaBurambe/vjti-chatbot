import gradio as gr
import os
import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load CSV FAQs
df = pd.read_csv('vjti_faqs.csv')
df.dropna(inplace=True)
docs = [Document(page_content=row['Answer'], metadata={"question": row['Question']}) for _, row in df.iterrows()]

# Create embeddings and FAISS vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up Gemini model and RAG chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 🧠 Smart chatbot logic: RAG first, then fallback to full LLM
def ask_chatbot(query, history=None):
    history = history or []

    # Step 1: Run similarity search first (manual RAG control)
    try:
        matches = vectorstore.similarity_search_with_score(query, k=1)
        if matches and matches[0][1] > 0.7:
            result = qa.invoke(query)
            answer = result.get("result", "").strip()
            if answer and len(answer.split()) > 5:
                return f"📌 *Based on our official FAQ:*\n\n{answer}"
    except Exception as e:
        print(f"[RAG error] {e}")

    # Step 2: Build prompt with personality for Gemini fallback
    system_prompt = (
        "You are the VJTI Helpdesk AI Assistant. "
        "Besides providing answers from the official FAQ, "
        "you are helpful, friendly, and capable of answering general queries "
        "about college life, academics, VJTI location, admission, weather, and more."
    )

    history_prompt = ""
    for turn in history:
        if turn.get("role") == "user":
            history_prompt += f"User: {turn['content']}\n"
        elif turn.get("role") == "assistant":
            history_prompt += f"Assistant: {turn['content']}\n"

    full_prompt = f"{system_prompt}\n\n{history_prompt}User: {query}\nAssistant:"

    try:
        response = llm.invoke(full_prompt)
        return f"💬 *AI Response:*\n\n{response.content.strip()}"
    except Exception as e:
        print(f"[LLM fallback error] {e}")
        return "⚠️ Sorry, I couldn't process your request right now."
source_q = matches[0][0].metadata.get("question", "")
return f"📌 *Based on:* \"{source_q}\"\n\n{answer}"


# 🧑‍💻 Gradio logic to handle message stream
def chatbot_response(user_input, history):
    history = history or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": "⏳ Typing..."})  # temp message
    time.sleep(0.5)
    response = ask_chatbot(user_input, history)
    history[-1]["content"] = response
    return history

# 🎨 Custom CSS styling
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

# 🚀 Gradio app layout
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🤖 VJTI Helpdesk Chatbot</h1>")

    chatbot = gr.Chatbot(
        type="messages",
        height=400,
        layout="bubble",
        avatar_images=("🧑", "🤖"),
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

 # 🌐 For Render or Hugging Face deployment
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
