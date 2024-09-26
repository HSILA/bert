from datasets import concatenate_datasets, load_dataset


def get_raw_datasets(wikipedia_dump="20231101.en"):
    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", wikipedia_dump, split="train")
    wiki = wiki.remove_columns(
        [col for col in wiki.column_names if col != "text"])

    assert bookcorpus.features.type == wiki.features.type
    raw_datasets = concatenate_datasets([bookcorpus, wiki])
    return raw_datasets
