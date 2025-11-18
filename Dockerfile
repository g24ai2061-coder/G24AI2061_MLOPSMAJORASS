# Dockerfile

# 1. Base Python image
FROM python:3.12-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4. Copy all project files (including app.py and savedmodel.pth)
COPY . .

# 5. Expose Flask port
EXPOSE 4000

# 6. Start the Flask app
CMD ["python", "app.py"]