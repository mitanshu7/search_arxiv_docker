# Use the Milvus base image
FROM milvusdb/milvus:v2.4.5

################################################################################

# Declare environment variables
ENV ETCD_USE_EMBED=true
ENV ETCD_DATA_DIR=/var/lib/milvus/etcd
ENV ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml
ENV COMMON_STORAGETYPE=local

# Copy config files
COPY ./embedEtcd.yaml /milvus/configs/embedEtcd.yaml
COPY ./user.yaml /milvus/configs/user.yaml

# Expose required ports
EXPOSE 19530 9091 2379 7860

################################################################################
# Install python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

COPY . .

# RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install -r requirements.txt

# Run script
CMD ["bash", "entrypoint.sh"]