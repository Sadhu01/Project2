from fastapi import FastAPI, File, UploadFile, Form
import pandas as pd
import zipfile
import io

app = FastAPI()

@app.post("/api/")
async def answer_question(
    question: str = Form(...),
    file: UploadFile = None
):
    if file:
        with zipfile.ZipFile(io.BytesIO(await file.read()), 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            if "extract.csv" in extracted_files:
                with zip_ref.open("extract.csv") as csv_file:
                    df = pd.read_csv(csv_file)
                    return {"answer": str(df["answer"].iloc[0])}
    
    return {"answer": "Generated response based on the question"}
