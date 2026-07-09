from transformers import AutoTokenizer, AutoModel

AutoTokenizer.from_pretrained(
    "distilbert-base-uncased",
    cache_dir="./models"
)

AutoModel.from_pretrained(
    "distilbert-base-uncased",
    cache_dir="./models"
)