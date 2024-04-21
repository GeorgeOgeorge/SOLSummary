import os
import re
import requests

import cv2
from pdf2image import convert_from_path

from backend.image_processor import ImageProcessor
from backend.text_summarizer import TextSummarizer


class ArticleSummarizer:
    doc: dict

    _EXTRACT_ABSTRACT: str = r"Abstract\.\s*(.*?)\x0c"
    _IS_TOPIC: str = r"(\d+(?:\.\d+)*\.\s\b\w{2}\w.*)(?=\n)"

    image_processor: ImageProcessor = ImageProcessor()
    text_summarizer: TextSummarizer = TextSummarizer()

    def summarize_file(self, file_path: str, output_path: str) -> None:
        """_summary_

        Args:
            file_path (str): _description_
            output_path (str): _description_
        """
        pages = convert_from_path(pdf_path=file_path, dpi=400, thread_count=10)

        page_info = [
            self.image_processor.mark_page_contours(page, index)
            for index, page in enumerate(pages)
        ]

        self.doc = self._document_as_dict(page_info)
        abstract = self.doc.pop("abstract")

        try:
            del self.doc['misc']
            del self.doc['Acknowledgement']
        except Exception:
            pass

        self.write_file(
            final_text=self._text_from_doc(),
            output_path=output_path,
            original_abstarct=abstract
        )

    def _document_as_dict(self, processed_pages_info) -> dict[str, str]:
        """_summary_

        Args:
            processed_pages_info (_type_): _description_

        Returns:
            dict[str, str]: _description_
        """
        document_dict = {"misc": ""}
        current_topic = None

        for page in processed_pages_info:
            page_path = page["image_path"]
            marked_image = cv2.imread(page_path)

            for contour in page["contours_coordinates"]:
                extracted_text = self.text_summarizer._extract_contour_text(marked_image, contour)

                if "Abstract. " in extracted_text:
                    text_search_result = re.search(self._EXTRACT_ABSTRACT, extracted_text, re.DOTALL | re.IGNORECASE)
                    document_dict["abstract"] = self.text_summarizer._preprocess_text(text_search_result.group(1))
                elif re.match(self._IS_TOPIC, extracted_text):
                    current_topic = extracted_text.replace("\n", "").replace("\x0c", "")
                    document_dict[current_topic] = ""
                else:
                    x_start, x_end = contour[0][0], contour[1][0]
                    processed_text = self.text_summarizer._preprocess_text(extracted_text)

                    if current_topic and x_end - x_start > 2000:
                        document_dict[current_topic] = f"{document_dict[current_topic]} {processed_text}"
                    else:
                        document_dict["misc"] = f'{document_dict["misc"]} {processed_text}'

            os.remove(os.path.abspath(page_path))

        return document_dict

    def _text_from_doc(self) -> str:
        """_summary_

        Returns:
            list[str]: _description_
        """
        final_text = " ".join(
            text
            for topico, text in self.doc.items()
            if any(sub.upper() in topico.upper() for sub in ("intro", "conclu", 'resul'))
        )

        return final_text

    def write_file(
        self,
        final_text: str,
        output_path: str,
        original_abstarct: str | None = "",
    ) -> None:
        """_summary_

        Args:
            temp_text_list (list[str]): _description_
            output_path (str): _description_
            original_abstarct (str | None, optional): _description_. Defaults to " ".
        """
        prompt = f"""
            please create a summary with 150-200 words, with the following content,
            only return result text, no introduction: {final_text}
        """
        print("começando resumo")
        final_text = requests.post(
            "http://localhost:6969/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        ).json().get("response", "")
        print("terminou resumo")

        with open(f"{output_path}/arquivo.txt", "w") as file:
            file.write("-"*20 + " original abstract " + "-"*20 + "\n")
            file.write(f"{original_abstarct} \n\n\n")
            file.write("-"*20 + " creted abstract " + "-"*20 + "\n")
            file.write(f"{final_text} \n\n\n")
