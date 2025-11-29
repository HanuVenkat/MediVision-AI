Medivision AI - Mini Project (ready-to-run scaffold)
==================================================

Contents:
- backend/: FastAPI backend (simulated AI replies). SQLite DB preloaded with sample images.
- frontend/: Streamlit frontend to upload image + voice WAV and call backend.
- requirements.txt: Python packages used (install into a virtualenv).
- .env.example: Example env file (put your GROQ_API_KEY here if using real AI).

How to run (locally):
1. Create & activate a virtualenv:
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # macOS / Linux

2. Install requirements:
   pip install -r requirements.txt

3. Start backend (in project root):
   uvicorn backend.brain_of_the_doctor:app --reload

4. Start frontend (in another terminal):
   streamlit run frontend/app.py

Notes:
- The database (medivision.db) is created inside backend/ and contains 5 sample records.
- The backend enforces a max of 6 records. If you reach the limit, delete records from the DB file or via code.

Security & next steps:
- Replace the simulated AI response with a real Groq or other model call (set GROQ_API_KEY in a .env file).
- Add FFmpeg and PortAudio on your machine if you plan to convert or record audio directly.
