FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (app.py, models, static)
COPY . .

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Start FastAPI using Uvicorn on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
