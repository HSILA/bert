from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
load_dotenv(override=True)

def get_raw_datasets(wikipedia_dump="20231101.en", hub_path=None):
    bookcorpus = load_dataset(
        "bookcorpus", split="train", trust_remote_code=True)
    wiki = load_dataset("wikipedia", wikipedia_dump, split="train")
    wiki = wiki.remove_columns(
        [col for col in wiki.column_names if col != "text"])

    assert bookcorpus.features.type == wiki.features.type
    raw_datasets = concatenate_datasets([bookcorpus, wiki])
    if hub_path:
        raw_datasets.push_to_hub(hub_path)
    return raw_datasets
