FROM python:3.12-slim
# Install MySQL client for health checks
RUN apt-get update && apt-get install -y default-mysql-client && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy the application into the container.
COPY . /app
# Install the application dependencies.
RUN pip install -r requirements.txt
EXPOSE 5050
# Wait for database and start the application (without migrations and seed)
CMD ["sh", "-c", "\
 echo 'Waiting for database...' && \
 while ! mysqladmin ping -h${DB_HOST:-agent-question-db} -P${DB_PORT:-3306} -u${DB_USER:-root} -p${MYSQL_ROOT_PASSWORD} --silent; do \
 echo 'Database not ready, waiting...' && sleep 2; \
 done && \
 echo 'Database is ready!' && \
 echo 'Starting FastAPI application...' && \
 fastapi run app/app.py --port 5050 --host 0.0.0.0"]