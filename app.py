import gradio as gr
from query_rag import query_rag

def answer_question(question):
    if not question:
        return "Ask me anything about work so far!"
    return query_rag(question)

# Settin up gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask about your monthly review PDFs..."),
    outputs=gr.Textbox(),
    title="RAG Assistant for work",
    description="Ask questions about your monthly review PDFs and get answers! Use this for resume building ;)"
)

if __name__ == "__main__":
    iface.launch()
