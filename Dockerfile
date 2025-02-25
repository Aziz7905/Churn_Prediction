FROM python:3.11

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for the FastAPI apps and MLflow
EXPOSE 8000 8001 5002

# Run both FastAPI applications and MLflow server when the container starts
CMD /bin/sh -c "
  mkdir -p mlruns &&
  uvicorn app:app --host 0.0.0.0 --port 8000 & 
  uvicorn app1:app --host 0.0.0.0 --port 8001 & 
  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file://$(pwd)/mlruns --host 0.0.0.0 --port 5002 & 
  wait
"
