FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_chatbot_app.py .
COPY swield_catalog.html .
COPY qabooklet.txt .

EXPOSE 7860

CMD ["python", "flask_chatbot_app.py"]