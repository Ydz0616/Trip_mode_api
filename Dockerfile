# Use the official Python base image
FROM python:3.8-slim

# Set environment variables to prevent Python from writing pyc files and ensure output is shown in real-time
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the FastAPI app port
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
