
build-container:
	@echo "=> Building Docker image..."
	docker build -t mountain-ner-api .
	@echo "=> Cleaning up old container if it exists..."
	-docker stop ner-service 2>/dev/null || true
	-docker rm ner-service 2>/dev/null || true
	@echo "=> Starting new container..."
	docker run -d -p 8080:8080 --name ner-service mountain-ner-api
	@echo "=> Service is running on http://localhost:8080"

stop-container:
	@echo "=> Stopping and removing container 'ner-service'..."
	-docker stop ner-service 2>/dev/null || true
	-docker rm ner-service 2>/dev/null || true
	@echo "=> Container stopped and removed."