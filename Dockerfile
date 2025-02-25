FROM python:3.11

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for the FastAPI apps and MLflow
EXPOSE 8000 8001 5002

# Create a shell script to run the necessary commands
RUN echo '#!/bin/bash\n\
mkdir -p mlruns && \n\
uvicorn app:app --host 0.0.0.0 --port 8000 & \n\
uvicorn app1:app --host 0.0.0.0 --port 8001 & \n\
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file://$(pwd)/mlruns --host 0.0.0.0 --port 5002 & \n\
wait' > start.sh && chmod +x start.sh

# Run the shell script
CMD ["./start.sh"]
