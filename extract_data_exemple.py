import json
import os

from backend.file_summarizer import ArticleSummarizer

TEMP_FILE_PATH = "./temp/"
summarizer = ArticleSummarizer()

articles_data = {
    file[:-4]: summarizer.summarize_file(file_path=f"{TEMP_FILE_PATH}{file}", output_path="")
    for file in os.listdir(TEMP_FILE_PATH)
    if "pdf" in file
}

for file_name, article_data in articles_data.items():
    with open(f"{TEMP_FILE_PATH}{file_name}.json", 'w', encoding='utf-8') as arquivo_json:
        json.dump(article_data, arquivo_json, indent=4, ensure_ascii=False)
