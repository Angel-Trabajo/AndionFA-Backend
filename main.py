from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from src.routes.routes_config import router as router_config
from src.routes.routes_engine import router as router_engine
from src.routes import peticiones
from src import engine_manager


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    peticiones.initialize_mt5()



load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos (GET, POST, etc.)
    allow_headers=["*"],  # Headers permitidos
)
app.include_router(router_config, prefix="/config", tags=["Configuracion"])
app.include_router(router_engine, prefix="/engine", tags=["Engine"])



@app.get("/")
async def root():
    peticiones.initialize_mt5()
    return {"message": "Hola mundo"}


if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)

#./env/Scripts/activate
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload