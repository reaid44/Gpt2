# app.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr
import torch

MODEL_NAME = "distilgpt2"

# চেষ্টা করবে যদি GPU থাকে তখন GPU ব্যবহার করতে
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    framework="pt"
)

SYSTEM_PROMPT = "তুমি এখন সহায়ক বাংলা চ্যাটবট। সংক্ষিপ্ত এবং পরিষ্কার উত্তর দাও।"

def generate_reply(user_input, max_length=150, temperature=0.7):
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nBot:"
    out = generator(prompt, max_length=max_length, temperature=temperature, do_sample=True, num_return_sequences=1)
    text = out[0]["generated_text"]
    # প্রম্পট অংশ বাদ দিয়ে শুধু উত্তরের অংশ রাখি
    reply = text.split("Bot:")[-1].strip()
    return reply

with gr.Blocks() as demo:
    gr.Markdown("## Bengali Chatbot (DistilGPT2 fallback)")
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="তুমি লিখো (বাংলায় লিখতে পারবে)", placeholder="এখানে মেসেজ লিখো...", lines=4)
            submit = gr.Button("Send")
        with gr.Column(scale=1):
            max_len = gr.Slider(minimum=50, maximum=512, value=150, step=50, label="Max tokens")
            temp = gr.Slider(minimum=0.1, maximum=1.2, value=0.7, step=0.1, label="Temperature")
    chat = gr.Chatbot()
    def submit_fn(message, ml, t, history):
        reply = generate_reply(message, max_length=ml, temperature=t)
        history = history + [[message, reply]]
        return "", ml, t, history
    submit.click(fn=submit_fn, inputs=[txt, max_len, temp, chat], outputs=[txt, max_len, temp, chat])
    txt.submit(fn=submit_fn, inputs=[txt, max_len, temp, chat], outputs=[txt, max_len, temp, chat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
