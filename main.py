from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define the expected input structure
class UserInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(user_input: UserInput):
    return {"response": f"You said: {user_input.message}"}