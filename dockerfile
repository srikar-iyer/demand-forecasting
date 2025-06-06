FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/data/input /app/data/output /app/static/images

# Copy static files first
COPY static/ /app/static/

# Copy application code
COPY *.py /app/

# Make port 7860 available (for Gradio)
EXPOSE 7860

# Create placeholder image files to ensure the directory has write permissions
RUN touch /app/static/images/placeholder.png && \
    chmod 777 /app/static/images -R

#  standalone_retail_forecast_ui.py was working retail_forecast.py not working
CMD ["python", "retail_forecast.py"]