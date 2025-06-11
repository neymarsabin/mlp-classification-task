import kagglehub
import os

os.makedirs("datasets", exist_ok=True)
base_path = "./datasets"

kaggle_locations = [
    {
        "name": "Bird",
        "url": "klu2000030172/birds-image-dataset"
    },
    {
        "name": "Car",
        "url": "kshitij192/cars-image-dataset"
    },
    {
        "name": "Elephant",
        "url": "vivmankar/asian-vs-african-elephant-image-classification"
    },
    {
        "name": "Flower",
        "url": "marquis03/flower-classification"
    },
    {
        "name": "Human",
        "url": "snmahsa/human-images-dataset-men-and-women"
    }
]

for location in kaggle_locations:
    dataset_name = location["name"]
    dataset_url = location["url"]
    save_path = os.path.join(base_path, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading {dataset_name} dataset...")
    path = kagglehub.dataset_download(dataset_url)

    # copy the path directory to the save_path directory
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            os.rename(item_path, os.path.join(save_path, item))
        else:
            os.rename(item_path, os.path.join(save_path, item))
    print(f"{dataset_name} dataset extracted to: {save_path}")
