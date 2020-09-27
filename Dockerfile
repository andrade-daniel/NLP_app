FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && \
apt-get upgrade -y && \
apt-get install -y git && \
pip3 install -r requirements.txt && \
pip3 install git+https://github.com/LIAAD/yake && \
python3 -m spacy download pt_core_news_sm && \
python3 -m spacy download en_core_web_sm

COPY . .

CMD streamlit run --server.enableCORS false app.py --server.port $PORT