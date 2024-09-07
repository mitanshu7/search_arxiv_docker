#!/bin/bash

# Run milvus in background
milvus run standalone > /dev/null 2>&1 &

# Wait for Milvus to be healthy and ready
echo "Waiting for Milvus to start..."
until curl -sf http://localhost:9091/healthz; do
    sleep 2
done
echo "Milvus is running."

# Prepare milvus
python3 prepare_milvus.py

echo "Prepared Milvus, Running app..."

# Start the Gradio app (app.py)
python3 app.py

