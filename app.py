from flask import Flask, render_template, request
from groq import Groq

app = Flask(__name__)

# তোমার Groq API Key এখানে বসাও
client = Groq(api_key="YOUR_GROQ_API_KEY")

@app.route("/", methods=["GET", "POST"])
def home():
    user_input = ""
    ai_response = ""
    
    if request.method == "POST":
        user_input = request.form.get("user_message")
        
        try:
            # Groq API-তে কল পাঠানো হচ্ছে
            completion = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[
                    {"role": "user", "content": user_input}
                ]
            )
            ai_response = completion.choices[0].message.content
        except Exception as e:
            ai_response = f"Error: {str(e)}"

    return render_template("index.html", prompt=user_input, response=ai_response)

if __name__ == "__main__":
    app.run(debug=True)
