# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy current directory's contents into /app in the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set default command
CMD ["python", "pipeline.py"]