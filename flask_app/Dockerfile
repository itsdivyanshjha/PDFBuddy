# Base image for Flask application
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Set working directory
WORKDIR /app

# Copy only requirements to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose Flask port
EXPOSE 5000

# Healthcheck (optional but recommended)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:5000/ || exit 1

# Command to start the Flask application
CMD ["python", "app.py"]
