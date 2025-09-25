from inference_endpoint.dataset_manager.dataloader import HFDataLoader, PickleReader


def test_ds_pickle_reader(ds_pickle_reader):
    ds_pickle_reader.load()
    assert ds_pickle_reader.num_samples() == 5
    data_item = ds_pickle_reader.load_sample(0)
    assert isinstance(data_item, dict)
    print(data_item["dataset"])
    print(data_item["ground_truth"])
    print(data_item["ref_accuracy"])
    assert "dataset" in data_item and data_item["dataset"] == "livecodebench"
    assert "ground_truth" in data_item and data_item["ground_truth"] == "3154"
    assert "ref_accuracy" in data_item and data_item["ref_accuracy"] == 100.0


def test_ds_pickle_reader_unique_dataset(ds_pickle_reader):
    ds_pickle_reader.load()
    unique_datasets = set()
    for i in range(ds_pickle_reader.num_samples()):
        samples = ds_pickle_reader.load_sample(i)
        unique_datasets.add(samples["dataset"])
    assert len(unique_datasets) == 5


def test_custom_parser_pickle_reader(ds_pickle_dataset_path):
    def parser(row):
        # custom parser to only return dataset and text_input
        return {"dataset": row["dataset"], "text_input": row["text_input"]}

    data_loader = PickleReader(ds_pickle_dataset_path, parser=parser)
    data_loader.load()
    # check number of samples
    assert data_loader.num_samples() == 5
    # check first sample
    samples = data_loader.load_sample(0)

    # check columns that were not requested are not present
    assert "ref_output" not in samples and "metric" not in samples
    # check columns that were requested are present
    assert "dataset" in samples and "text_input" in samples
    # check order or rows - zeroth row should be livecodebench
    assert samples["dataset"] == "livecodebench"


def test_hf_squad_dataset(hf_squad_dataset):
    hf_squad_dataset.load()
    assert hf_squad_dataset.num_samples() == 50
    sample = hf_squad_dataset.load_sample(0)
    assert all(k in sample for k in ["id", "title", "context", "question", "answers"])
    assert sample["title"] == "Egypt"


def test_custom_parser_hf_squad_dataset(hf_squad_dataset_path):
    def parser(row):
        return {
            "title": row["title"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"],
        }

    dataloader = HFDataLoader(hf_squad_dataset_path, parser=parser, format="arrow")
    dataloader.load()
    assert dataloader.num_samples() == 50
    sample = dataloader.load_sample(0)
    assert "id" not in sample
    assert all(k in sample for k in ["title", "context", "question", "answers"])
    assert sample["title"] == "Egypt"
