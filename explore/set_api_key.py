import os

OPENAI_API_KEY = "add-your-secret-key"

def set_key():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("[INFO] OPENAI_API_KEY is set for this session.")

def get_key():
    #return OPENAI_API_KEY
    return None