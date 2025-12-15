import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

base_url = "https://openrouter.ai/api/v1"
api_key = os.getenv("OPENROUTER_API_KEY")

deepseek = OpenAI(base_url=base_url, api_key=api_key)

model_name = "mistralai/devstral-2512:free"


def chat(message, history, ebook_path):

    reader = PdfReader(ebook_path)

    if not ebook_path:
        return "Please upload a pdf file to start chatting."
    
    full_book = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_book += text

    with open("book/summary.txt", "w", encoding="utf-8") as f:
        f.write(full_book)

    with open("book/summary.txt","r",encoding="utf-8") as f:
        summary = f.read()

    system_prompt = f"You are acting a book summarizer. You are answering questions on a website, \
    particularly questions related to book \
    You are given a summary of the book which you can use to answer questions. \
    Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
    If you don't know the answer, say so."

    system_prompt += f"\n\n## Summary:\n{summary}\n\n"
    system_prompt += f"With this context, please chat with the user."

    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user","content": message}]
    response = deepseek.chat.completions.create(model=model_name, messages= messages)
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.File(label="Upload an ebook(pdf)", file_types=[".pdf"], type="filepath")
    ]
)

demo.launch()