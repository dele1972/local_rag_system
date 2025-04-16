# app/ui.py
import gradio as gr
from app.loader import load_documents_from_path
from app.vectorstore import build_vectorstore
from app.rag import build_qa_chain

#available_models = ["llama3", "mistral", "phi"]
available_models = ["llama3.2", "mistral", "deepseek-r1"]

qa_chain = None


def setup_qa(model, doc_path):
    global qa_chain
    docs = load_documents_from_path(doc_path)
    vs = build_vectorstore(docs, model)
    qa_chain = build_qa_chain(vs, model)
    return f"Modell '{model}' wurde geladen und Dokumente aus '{doc_path}' wurden verarbeitet."


def ask_question(question):
    if qa_chain is None:
        return "Bitte zuerst Modell und Dokumente laden."
    # return qa_chain.run(question)
    return qa_chain.invoke({"query": question})


def start_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Lokales RAG mit Ollama")
        model = gr.Dropdown(choices=available_models, label="Modellwahl")
        doc_path = gr.Textbox(label="Pfad zur Dokumentenbasis")
        load_btn = gr.Button("Laden")
        status = gr.Textbox(label="Status")

        question = gr.Textbox(label="Frage")
        answer = gr.Textbox(label="Antwort")
        ask_btn = gr.Button("Frage stellen")

        load_btn.click(setup_qa, inputs=[model, doc_path], outputs=status)
        ask_btn.click(ask_question, inputs=question, outputs=answer)

    # demo.launch()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)