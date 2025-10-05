import os
import json
import re
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Configure the Gemini API ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found. Chatbot will not work.")
else:
    genai.configure(api_key=api_key)

# --- Load Knowledge Bases (Schemes and Farmer Data) ---
knowledge_base = None
knowledge_base_string = ""
try:
    with open('schemes_data.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
        knowledge_base_string = json.dumps(knowledge_base, indent=2)
    print("Successfully loaded schemes_data.json")
except Exception as e:
    print(f"Error loading schemes_data.json: {e}")

farmer_df = None
try:
    FARMER_DATA_FILE = 'enriched_soil.csv'
    farmer_df = pd.read_csv(FARMER_DATA_FILE)
    if 'rtc_number' not in farmer_df.columns:
        raise Exception("Column 'rtc_number' not found in farmer data file.")
    farmer_df['cleaned_rtc'] = farmer_df['rtc_number'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()
    print(f"Successfully loaded farmer data from {FARMER_DATA_FILE}")
    print(f"Total farmers in database: {len(farmer_df)}")
except Exception as e:
    print(f"Error loading farmer data: {e}")
    farmer_df = None

# --- AI Prompt Template ---
prompt_template = """
You are 'Sahayaka Mitra' (ಸಹಾಯಕ ಮಿತ್ರ), a helpful AI chatbot for Indian farmers. 
Your goal is to answer user questions about government agricultural schemes.

**RULES:**
1.  You must answer questions based ONLY on the information provided in the 'SCHEME INFORMATION' section below. Do not use any external knowledge.
2.  First, detect if the user's question is in English or Kannada.
3.  You MUST respond clearly and concisely in the both language English and Kannada to the user's question.
4.  If the user's question is unclear or not related to the provided schemes, politely say that you can only answer questions about the listed agricultural schemes.
5. If the user asks questions about topics outside agricultural schemes, politely inform them that you can only assist with questions related to the provided schemes.
6. If the user needs help with other agricultural schemes give them relevant details about the scheme and how to apply for it by searching online.
7. Always encourage users to visit official government websites or contact local agricultural offices for the most accurate and up-to-date information.

---
**SCHEME INFORMATION:**
{knowledge_base}
---

Now, based on the rules and information above, answer the user's question.

User Question: {user_question}
"""

# --- Health Check Route ---
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "knowledge_base_loaded": knowledge_base is not None,
        "farmer_data_loaded": farmer_df is not None,
        "api_key_configured": api_key is not None
    })

# --- Main UI Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Farmer Data Route ---
@app.route('/get_farmer_data', methods=['POST'])
def get_farmer_data():
    if farmer_df is None:
        return jsonify({'error': 'Farmer data is not available on the server.'}), 500

    data = request.get_json()
    rtc_number = data.get('rtc')
    
    if not rtc_number:
        return jsonify({'error': 'RTC number is required.'}), 400

    cleaned_rtc_input = re.sub(r'[^a-zA-Z0-9]', '', str(rtc_number)).lower()
    result_row = farmer_df[farmer_df['cleaned_rtc'] == cleaned_rtc_input]

    if not result_row.empty:
        farmer_data = result_row.iloc[0].to_dict()
        # Remove 'cleaned_rtc' if present
        farmer_data.pop('cleaned_rtc', None)
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        for key, value in farmer_data.items():
            if pd.isna(value):
                farmer_data[key] = None
            elif hasattr(value, 'item'):
                farmer_data[key] = value.item()
        
        return jsonify(farmer_data)
    else:
        return jsonify({'error': 'Farmer with this RTC number not found.'}), 404

# --- Chatbot Route ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('message')
    
    if not user_question:
        return jsonify({'reply': 'Invalid request. Please provide a message.'}), 400

    if not knowledge_base_string:
        return jsonify({'reply': 'Knowledge base not loaded. Please contact administrator.'}), 500
    
    if not api_key:
        return jsonify({'reply': 'Chatbot API is not configured. Please contact administrator.'}), 500

    try:
        final_prompt = prompt_template.format(
            knowledge_base=knowledge_base_string,
            user_question=user_question
        )
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(final_prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({'reply': 'Sorry, I encountered an error. Please try again later.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Sahayaka Mitra Farmer Portal")
    print("="*50)
    print(f"Schemes loaded: {knowledge_base is not None}")
    print(f"Farmer data loaded: {farmer_df is not None}")
    print(f"API configured: {api_key is not None}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)