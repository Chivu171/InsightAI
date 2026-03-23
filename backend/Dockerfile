# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
# Clean uninstall and fresh install of LangChain 0.3
RUN pip uninstall -y langchain langchain-community langchain-core langchain-experimental langchain-huggingface langchain-text-splitters
RUN pip install --no-cache-dir langchain==0.3.7 langchain-community==0.3.5 langchain-core==0.3.15 langchain-experimental==0.3.3 langchain-huggingface==0.1.1 langchain-text-splitters==0.3.2
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import langchain; print('LangChain version:', langchain.__version__); print('LangChain path:', langchain.__path__)"
RUN find /usr/local/lib/python3.10/site-packages -name "parent_document_retriever.py"

# Copy the rest of the backend code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
