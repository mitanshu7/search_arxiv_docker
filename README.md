# Run milvus db app to search vectors for similarity in arxiv abstract embeddings

## Usage:
1. `podman build -t search_arxiv_docker .`
2. `podman run -p 7860:7860 search_arxiv_docker`
