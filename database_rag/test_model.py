from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

print("Available Models for your API Key:")
try:
    models = client.models.list()
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"Error: {e}")