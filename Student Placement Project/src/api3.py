from fastapi import FastAPI

app = FastAPI(title="Student Placement API")


@app.get("/")
def read_root():
    return {"message": "Student Placement API is running"}