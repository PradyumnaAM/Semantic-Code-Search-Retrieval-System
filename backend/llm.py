from openai import OpenAI, APIConnectionError

ollama = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen3:4b"

def ask_llm(question: str) -> str:
    try:
        response = ollama.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI software engineering assistant.
                    your job is to find for bugs, explain files or code snippets, suggest changes when the user asks you to.
                    Never give out a wrong answer if you dont know. If you are unsure, dont hallucinate.
                    """
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        
        return response.choices[0].message.content
    except APIConnectionError as error:
        return "Unable to connect to Ollama. Please make sure Ollama is running."
    