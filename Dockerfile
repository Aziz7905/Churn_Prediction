FROM python:3.11

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans l'image Docker
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Lancer l'application FastAPI en prenant en compte la structure du projet
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

