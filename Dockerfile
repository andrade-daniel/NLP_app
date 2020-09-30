FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && \
# apt-get upgrade -y && \
apt-get install -y git && \
pip3 install -r requirements.txt

# EXPOSE 8080
EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

COPY . /app

# CMD streamlit run --server.enableCORS false app.py --server.port $PORT
CMD streamlit run --server.port 80 --server.enableCORS false app.py