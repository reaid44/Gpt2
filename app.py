from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)

MODEL_NAME = "distilgpt2"
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
    reply = text.split("Bot:")[-1].strip()
    return reply

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_input = data.get("input", "")
    reply = generate_reply(user_input)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
