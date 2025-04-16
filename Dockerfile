# Use an official Python runtime as a parent image.
# Use Python 3.12.4-slim or a comparable variant.
FROM python:3.12.4-slim

# Set environment variables to prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in the container.
WORKDIR /app

# Install system dependencies. (You may add more as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (to leverage Docker's cache)
COPY requirements.txt /app/

# Install Python dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project code into the container.
COPY . /app/

# Expose the port FastAPI will run on.
EXPOSE 8000

# Command to run the API using Uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
