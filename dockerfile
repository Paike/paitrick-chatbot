# temp stage
FROM python:3.12.2-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

RUN mkdir -p "model"

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); model.save('model')"

COPY ["public", "/app/public"]
COPY [".chainlit/config.toml", "/app/.chainlit/config.toml"]
COPY ["chatbot.py", "prompts.py", "chainlit.md", "/app/"]

# final stage
FROM python:3.12.2-slim AS deploy 

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY --from=builder /app ./

ENV PATH="/opt/venv/bin:$PATH"

CMD ["chainlit", "run", "chatbot.py"] 