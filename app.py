from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from main import rag_chain  # ✅ Importing RAG Chain from main.py

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def ask_question_page(request: Request):
    """Renders the UI for users to ask questions."""
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

@app.post("/", response_class=HTMLResponse)
async def get_answer(request: Request, question: str = Form(...)):
    """Handles user questions and returns an answer."""
    answer = rag_chain(question)
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
