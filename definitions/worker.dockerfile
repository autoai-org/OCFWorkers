FROM huggingface/transformers-pytorch-gpu:latest

COPY src /app
COPY requirements.txt /app/requirements.txt
ENV HF_HOME=/cache

WORKDIR /app

RUN pip install -U -r requirements.txt && pip install -U transformers && pip install -U huggingface-hub