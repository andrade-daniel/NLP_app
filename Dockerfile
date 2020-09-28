FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && \
# apt-get upgrade -y && \
apt-get install -y git && \
pip3 install -r requirements.txt

EXPOSE 8080

COPY . .

# CMD streamlit run --server.enableCORS false app.py --server.port $PORT
CMD streamlit run --server.port 8080 --server.enableCORS false app.py