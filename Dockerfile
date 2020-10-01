FROM python:3.7-slim

COPY . /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && \
apt-get install -y git && \
pip3 install -r requirements.txt

EXPOSE 80

RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

WORKDIR /app

# CMD streamlit run --server.port 80 --server.enableCORS false app.py
CMD streamlit run app.py