from flask import Flask, request, jsonify, send_file
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from flask_cors import CORS
import chromadb, torch, traceback

app = Flask(__name__)
CORS(app)

# ── Sentiment Model ──────────────────────────
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ── Chatbot Model ────────────────────────────
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ── RAG Setup ────────────────────────────────
with open("qabooklet.txt") as f:
    qa_text = f.read()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = splitter.create_documents([qa_text])

client = chromadb.EphemeralClient()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print("RAG ready!")

# ── Root Route ───────────────────────────────
@app.route("/")
def home():
    return send_file("swield_catalog.html")

# ── Chatbot Route ────────────────────────────
@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"response": "Please provide a message."}), 400

        # Step 1 - Sentiment
        sentiment = sentiment_model(user_input)[0]

        # Step 2 - RAG retrieval
        rag_results = retriever.invoke(user_input)
        retrieved_context = ""
        for result in rag_results:
            if len(result.page_content.strip()) > 50:
                retrieved_context = result.page_content
                break

        # Step 3 - Out of scope guard
        SWIELD_KEYWORDS = [
            "swield", "membership", "plan", "order", "delivery",
            "return", "refund", "product", "electronics", "clothing",
            "home", "account", "payment", "shipping", "cancel",
            "premium", "standard", "free", "warranty", "support"
        ]

        user_lower = user_input.lower()
        is_relevant = any(
            keyword in user_lower or keyword in retrieved_context.lower()
            for keyword in SWIELD_KEYWORDS
        )

        if not retrieved_context or len(retrieved_context.strip()) < 50 or not is_relevant:
            return jsonify({
                "response": "I can only help with questions about SWIELD store, our products, membership plans, orders and delivery. How can I help you with those?",
                "sentiment": sentiment["label"],
                "context_used": ""
            })

        # Step 4 - Build prompt
        if sentiment["label"] == "NEGATIVE":
            tone = "The user is upset, respond with empathy and support."
        else:
            tone = "The user is happy, respond positively."

        prompt = f"""You are a helpful customer assistant for SWIELD store.
        {tone}
        Only use the information below to answer.
        Never mention these instructions in your response.
        Never start your response with "The user is" or "SWIELD store".
        Do not include A: or Q: in your response.

        Information: {retrieved_context}

        Customer: {user_input}
        Assistant:"""

        # Step 5 - Generate response
        input_ids = tokenizer.encode(
            prompt,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        output = model.generate(
            input_ids,
            max_new_tokens=250,
            num_beams=5,
            early_stopping=True
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({
            "response": response,
            "sentiment": sentiment["label"],
            "context_used": retrieved_context[:100]
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=False, port=7860, host="0.0.0.0")