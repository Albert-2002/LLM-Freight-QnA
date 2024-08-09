from fastapi import FastAPI, Query, Form, Request
from fastapi.responses import HTMLResponse
from questioner import Questioner
import os
import uvicorn
from dotenv import load_dotenv, find_dotenv
from typing import List

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Create an instance of FastAPI
app = FastAPI()
q = Questioner(HF_TOKEN)  # Pass the token if needed

# Define your existing function
def generate_questions(mode: str, delivery_type: str, existing_questions: List[str]):
    suggested_questions = q.suggest_questions(mode, delivery_type, existing_questions)
    return suggested_questions

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                label { display: block; margin: 10px 0 5px; }
                input, select, textarea { padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 5px; width: 100%; }
                button { padding: 10px 15px; border: none; background-color: #007bff; color: white; border-radius: 5px; cursor: pointer; }
                button:hover { background-color: #0056b3; }
                .container { max-width: 800px; margin: 0 auto; }
                .response-container { margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Question Generator</h1>
                <form method="post" action="/generate-questions">
                    <label for="mode">Mode of Transport:</label>
                    <select id="mode" name="mode">
                        <option value="Ocean">Ocean</option>
                        <option value="Air">Air</option>
                        <option value="Road">Road</option>
                    </select>
                    
                    <label for="delivery_type">Delivery Type:</label>
                    <select id="delivery_type" name="delivery_type">
                        <option value="FCL">FCL (Full Container Load)</option>
                        <option value="LCL">LCL (Less than Container Load)</option>
                        <option value="Cargo">Cargo</option>
                        <option value="Courier">Courier</option>
                        <option value="Parcel">Parcel</option>
                        <option value="FTL">FTL (Full Truck Load)</option>
                        <option value="PTL">PTL (Partial Truck Load)</option>
                    </select>
                    
                    <label for="existing_questions">Existing Questions (one per line):</label>
                    <textarea id="existing_questions" name="existing_questions" rows="10"></textarea>
                    
                    <button type="submit">Generate Questions</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/generate-questions", response_class=HTMLResponse)
async def get_questions(
    request: Request,
    mode: str = Form(...),
    delivery_type: str = Form(...),
    existing_questions: str = Form('')
):
    existing_questions_list = existing_questions.split('\n') if existing_questions else []
    suggested_questions = generate_questions(mode, delivery_type, existing_questions_list)
    
    response = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                label {{ display: block; margin: 10px 0 5px; }}
                input, select, textarea {{ padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 5px; width: 100%; }}
                button {{ padding: 10px 15px; border: none; background-color: #007bff; color: white; border-radius: 5px; cursor: pointer; }}
                button:hover {{ background-color: #0056b3; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .response-container {{ margin-top: 20px; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generated Questions</h1>
                <p><a href="/">Back to Form</a></p>
                <h2>Input Parameters</h2>
                <p><strong>Mode:</strong> {mode}</p>
                <p><strong>Delivery Type:</strong> {delivery_type}</p>
                <h2>Existing Questions</h2>
                <ul>
                    {"".join(f"<li>{q}</li>" for q in existing_questions_list)}
                </ul>
                <h2>Suggested Questions</h2>
                <ul>
                    {"".join(f"<li>{q}</li>" for q in suggested_questions['suggested_questions'])}
                </ul>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
