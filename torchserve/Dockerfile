# TorchServe Dockerfile for IMDb sentiment classification
# Using CPU version for better compatibility (especially on Mac M1)
FROM pytorch/torchserve:latest

# Set working directory
WORKDIR /home/model-server

# Copy model store and configuration
COPY torchserve/model-store /home/model-server/model-store
COPY torchserve/config.properties /home/model-server/config.properties

# Expose ports
# 8080: Inference API
# 8081: Management API
# 8082: Metrics API
EXPOSE 8080 8081 8082

# Start TorchServe
CMD ["torchserve", \
     "--start", \
     "--model-store=/home/model-server/model-store", \
     "--models=imdb-sentiment=imdb-sentiment.mar", \
     "--ts-config=/home/model-server/config.properties"]
