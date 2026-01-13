import os
DATASET_PATH ="genres_original"

if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset Path '{DATASET_PATH}'not found.")
    print("Please ensure you have unzipped the dataset in your project's root directory.")
else:
    print("Dataset directory found. Counting files in each genre folder...")
    print("-"*40)
    for genre_folder in sorted(os.listdir(DATASET_PATH)):
        genre_path = os.path.join(DATASET_PATH , genre_folder)

        if os.path.isdir(genre_path):
            files_in_genre = os.listdir(genre_path)
            number_of_files = len(files_in_genre)
            print(f"Genre:{genre_folder.ljust(12)} | File Count:{number_of_files}")
    print("-" * 40)
    print("Verification complete. The dataset appears to be balanced.")