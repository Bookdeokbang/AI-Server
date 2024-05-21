import uvicorn
from fastapi import FastAPI
from routers import predict_router, ocr_router

app = FastAPI()

app.include_router(predict_router)
app.include_router(ocr_router)

if __name__ == "__main__":
    uvicorn.run("main:app")