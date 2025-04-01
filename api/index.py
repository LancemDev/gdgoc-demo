from flask import Flask, request, jsonify, render_template
import os
import dotenv
import logging
from pinecone import Pinecone
from openai import OpenAI
from flask_cors import CORS

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ======= CONFIGURATION =======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "rag-demo"

# ======= SETUP =======
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def get_embedding(text):
    """Generates an embedding using OpenAI's Ada model"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieves top K relevant text chunks from Pinecone"""
    logger.info(f"Retrieving top {top_k} chunks for query: {query}")
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        
        include_metadata=True
    )
    
    chunks = [match.metadata["text"] for match in results.matches]
    logger.info(f"Retrieved {len(chunks)} relevant chunks")
    return chunks

def generate_answer(query, chunks):
    """Generates answer using GPT-4 based on retrieved story chunks"""
    logger.info(f"Generating answer for query: {query}")
    context = "\n".join(chunks)
    
    prompt = f"Using the provided story context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that extracts answers from given text."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        logger.info("Successfully generated answer")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

# ======= API ROUTES =======
@app.route('/', methods=['GET'])
def home():
    """Home route"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "RAG system is running"})

@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint to ask questions about the stored document"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        top_k = data.get('top_k', 3)  # Optional parameter with default value
        
        # Get relevant chunks
        chunks = retrieve_relevant_chunks(question, top_k)
        
        # Generate answer
        answer = generate_answer(question, chunks)
        
        response = {
            "question": question,
            "answer": answer,
            "context": chunks,  # Optionally include the chunks used
            "num_chunks": len(chunks)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chunks', methods=['POST'])
def get_relevant_chunks():
    """Endpoint to get relevant chunks for a query without generating an answer"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 3)
        
        chunks = retrieve_relevant_chunks(query, top_k)
        
        return jsonify({
            "query": query,
            "chunks": chunks,
            "num_chunks": len(chunks)
        })
    
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)