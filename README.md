\# Nutrition Chatbot API with Cosine Similarity



AI-powered nutrition assistant using Google Gemma-2-2b-it model with fuzzy ingredient matching.



\## Features



\- ✅ AI-powered calorie calculation

\- ✅ Fuzzy ingredient matching (cosine similarity) - handles typos!

\- ✅ Similar recipe recommendations

\- ✅ Workout plans (Beginner/Intermediate/Advanced)

\- ✅ Conversational AI (Gemma-2-2b-it)

\- ✅ RESTful API with FastAPI



\## Requirements



\- Python 3.8+

\- 16GB RAM (for running Gemma model)

\- HuggingFace account with API token



\## Installation



1\. Clone the repository:

```bash

git clone https://github.com/Chbaya7-Hamza/IAS.git

cd IAS

```



2\. Install dependencies:

```bash

pip install -r requirements.txt

```



3\. Create `.env` file:

```bash

HUGGINGFACE\_API\_KEY=your\_token\_here

```



Get your token from: https://huggingface.co/settings/tokens



\*\*Important:\*\* Accept Gemma license at https://huggingface.co/google/gemma-2-2b-it



4\. Run the API:

```bash

uvicorn main:app --reload --port 8001

```



\*\*Note:\*\* First run will download ~10GB Gemma model (takes 10-20 minutes).



\## API Endpoints



\- `GET /` - Health check

\- `POST /chat` - Conversational AI

\- `POST /calculate-calories` - Calculate meal calories with AI matching

\- `POST /analyze-meal` - Quick meal analysis

\- `POST /suggest-meal` - Meal suggestions by calories

\- `GET /workout-plan?level=beginner` - Get workout plans



\## Usage Examples



\### Calculate Calories

```bash

curl -X POST http://localhost:8001/calculate-calories \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"ingredients": "200g chicken breast, 150g brown rice, 100g broccoli"}'

```



\### Fuzzy Matching (with typos!)

```bash

curl -X POST http://localhost:8001/calculate-calories \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"ingredients": "180g chiken, 120g rize, 80g brocoli"}'

```



\### Chat

```bash

curl -X POST http://localhost:8001/chat \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"messages": \[{"role": "user", "content": "What should I eat for breakfast?"}]}'

```



\## Features



\### Cosine Similarity Matching

Uses `sentence-transformers` to match ingredients even with typos:

\- "chiken" → "chicken breast"

\- "rize" → "brown rice"

\- "brocoli" → "broccoli"



\### Nutrition Database

Contains 78+ foods with accurate calorie information per 100g.



\### AI Models

\- \*\*Main LLM:\*\* google/gemma-2-2b-it (10GB)

\- \*\*Embeddings:\*\* all-MiniLM-L6-v2 (90MB)



\## API Documentation



Once running, visit: http://localhost:8001/docs



\## Credits



Based on Kaggle notebook: Nutrition Chatbot with Cosine Similarity



\## License



MIT License

