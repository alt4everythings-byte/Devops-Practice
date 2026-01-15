# Lab: Serving ML Model using FastAPI

## Objective
Serve a machine learning model using FastAPI by creating a /predict REST endpoint
with input validation, writing test cases, containerizing the service, and verifying
predictions using curl.

---

## Project Structure
.
├── main.py
├── model.py
├── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md

---

## Steps to Execute (Without Docker)
1. Create virtual environment and activate it.
2. Install dependencies:
   pip install -r requirements.txt

3. Run FastAPI server:
   uvicorn main:app --reload

4. Test using curl:
   curl -X POST http://127.0.0.1:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"feature1":5,"feature2":10}'

---

## Steps to Execute (With Docker)
1. docker build -t fastapi-ml .
2. docker run -p 8000:8000 fastapi-ml
3. Test using curl (same as above)
