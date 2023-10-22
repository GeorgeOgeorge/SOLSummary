import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from numpy import ndarray
from pdf2image import convert_from_path
from PIL.PpmImagePlugin import PpmImageFile
from pytesseract import image_to_string
from transformers import BartForConditionalGeneration, BartTokenizer


@dataclass
class ArticleSummarizer:

    _model: BartForConditionalGeneration
    _tokenizer: BartTokenizer

    _MODEL_NAME: str = "facebook/bart-large-cnn"
    _IS_TOPIC = r"(\d+(?:\.\d+)*\.\s\b\w{2}\w.*)(?=\n)"
    _EXTRACT_ABSTRACT = r"Abstract\.\s*(.*?)\x0c"

    def __init__(self) -> None:
        self._model = BartForConditionalGeneration.from_pretrained(self._MODEL_NAME)
        self._tokenizer = BartTokenizer.from_pretrained(self._MODEL_NAME)

    def _preprocess_image(self, original_img: ndarray) -> ndarray:
        """Preprocesses an image for machine learning
            - set image to grayscale
            - blurs the image
            - performs image thresholding using gaussian adaptive threshold
            - dilates imagem 5 times

        Args:
            original_img (ndarray): The original image

        Returns:
            ndarray: The preprocessed image
        """
        image = cv2.cvtColor(original_img, code=cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=0)
        image = cv2.adaptiveThreshold(
            image,
            maxValue=255,
            C=30,
            blockSize=11,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
        )
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(15, 6))
        image = cv2.dilate(image, kernel=kernel, iterations=5)

        return image

    def _contours_intersect(
        self,
        img_hight: int,
        current_x: int,
        current_y: int,
        current_w: int,
        _: int,
        next_x: int,
        next_y: int,
        next_w: int,
        next_h: int
    ) -> bool:
        """checks if images contours intersect

        Args:
            img_hight (int): docuemnt image hight
            current_x (int): current image initial width
            current_y (int): current image inital hight
            current_w (int): current image final width
            _ (int): current image final hight
            next_x (int): next image final width
            next_y (int): next image inital hight
            next_w (int): next image final width
            next_h (int): next image final hight

        Returns:
            bool: whether the images contours interset or not
        """
        threshold = 0.002
        return (
            (abs((current_y / img_hight) - ((next_y + next_h) / img_hight))) < threshold
            and (current_x + current_w == next_x + next_w)
        )

    def _remove_nearby_contours(self, contours: list[ndarray], img_hight: int) -> list[ndarray]:
        """Removes nearby contours from a list of contours

        Args:
            contours (list[ndarray]): A list of contours
            img_hight (int): The height of the image

        Returns:
            list[ndarray]: A list of contours without nearby contours
        """
        contour_to_delete = []

        for index, contour in enumerate(contours):

            if index > 0 and self._contours_intersect(
                img_hight,
                *cv2.boundingRect(contour),
                *cv2.boundingRect(contours[index-1])
            ):
                contours[index] = np.concatenate((contours[index], contours[index-1]), axis=0)
                contour_to_delete.append(index-1)

        filtered_conturs = [contour for index, contour in enumerate(contours) if index not in contour_to_delete]

        return filtered_conturs

    def _extract_image_contours(self, original_img: ndarray) -> list[ndarray]:
        """Extracts contours from an image.

        Args:
            original_img (ndarray): The original image.

        Returns:
            list[ndarray]: A list of contours found in image.
        """
        contours_found = cv2.findContours(original_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_found[0] if len(contours_found) == 2 else contours_found[1]

        contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[1])
        sorted_contours = self._remove_nearby_contours(contours=contours, img_hight=original_img.shape[0])

        return sorted_contours

    def _write_contours(self, image: ndarray, contour: ndarray) -> list[tuple[int, int]]:
        """Writes contours to an image.

        Args:
            image (ndarray): The image to write the contours to.
            contour (ndarray): The contour to write.

        Returns:
            list[tuple[int, int]]: A list of the coordinates of the start and end points of the contour.
        """
        x_start, y_start, width, hight = cv2.boundingRect(contour)
        x_end, y_end = x_start + width, y_start + hight

        cv2.rectangle(image, pt1=(x_start, y_start), pt2=(x_end, y_end), color=(255, 0, 255), thickness=3)

        return [(x_start, y_start), (x_end, y_end)]

    def _mark_page_contours(
        self,
        page: PpmImageFile,
        page_index: int
    ) -> dict[str, str | list[list[tuple[int, int]]]]:
        """Marks the contours of a page.

        Args:
            page (PpmImageFile): The page to mark the contours of.
            page_index (int): The index of the page.

        Returns:
            dict[str, str | list[list[tuple[int, int]]]]:
                A dictionary with the image path and the contours coordinates.
        """
        page_name = f"{page_index}_page.jpg"
        page_file_location = f"./temp/{page_name}"
        processed_page_location = f"./temp/{page_index}.jpg"

        page.save(page_file_location, "JPEG")
        page_img = orignal_img = cv2.imread(page_file_location)

        page_img = self._preprocess_image(original_img=page_img)

        contours = self._extract_image_contours(original_img=page_img)

        contours_coordinates = [self._write_contours(image=orignal_img, contour=contour) for contour in contours]

        cv2.imwrite(processed_page_location, orignal_img)
        os.remove(os.path.abspath(page_file_location))

        return {"image_path": processed_page_location, "contours_coordinates": contours_coordinates}

    def _extract_contour_text(self, image: ndarray, contour: list[tuple[int, int]]) -> str:
        """Extracts text from a contour.

        Args:
            image (ndarray): The image to extract the text from.
            contour (list[tuple[int, int]]): The contour to extract the text from.

        Returns:
            str: The extracted text.
        """
        croped_img = image[contour[0][1]: contour[1][1], contour[0][0]: contour[1][0]]

        _, new_image = cv2.threshold(croped_img, thresh=120, maxval=255, type=cv2.THRESH_BINARY)

        return str(image_to_string(image=new_image, config="--psm 6"))

    def _preprocess_text(self, text: str) -> str:
        """removes special characters from string

        Args:
            text (str): string to be processed

        Returns:
            str: string with removed special characters
        """
        return text.replace("-\n", "").replace("\n", "").replace("\x0c", "") if text else ""

    def _build_document_dict(self, processed_pages_info):
        document_dict = {"misc": ""}
        current_topic = None

        for page in processed_pages_info:
            page_path = page["image_path"]
            marked_image = cv2.imread(page_path)

            for contour in page["contours_coordinates"]:
                extracted_text = self._extract_contour_text(marked_image, contour)

                if "Abstract. " in extracted_text:
                    text_search_result = re.search(self._EXTRACT_ABSTRACT, extracted_text, re.DOTALL | re.IGNORECASE)
                    processed_text = self._preprocess_text(text_search_result.group(1))
                    document_dict["abstract"] = processed_text
                elif re.match(self._IS_TOPIC, extracted_text):
                    extracted_text = extracted_text.replace("\n", "").replace("\x0c", "")
                    current_topic = extracted_text
                    document_dict[current_topic] = ""
                else:
                    x_start, x_end = contour[0][0], contour[1][0]

                    if current_topic and x_end - x_start > 2000:
                        processed_text = self._preprocess_text(extracted_text)
                        document_dict[current_topic] = f"{document_dict[current_topic]} {processed_text}"
                    else:
                        processed_text = self._preprocess_text(extracted_text)
                        document_dict["misc"] = f'{document_dict["misc"]} {processed_text}'

            os.remove(os.path.abspath(page_path))

        return document_dict

    def _split_text(self, text: str, max_tokens: int = 1000):
        tokens = self._tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

        if tokens.shape[1] <= max_tokens:
            return [text]

        split_point = len(text)//2

        part1 = text[:split_point]
        part2 = text[split_point:]

        return [part1] + self._split_text(part2)

    def _summary_text(self, input_text: str) -> str:
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self._model.generate(
            input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True
        )

        summary_text = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text

    def _build_document(self, content_dict: dict) -> dict:
        with open('arquivo.txt', 'w') as file:
            for topic, text in content_dict.items():
                file.write("-"*20 + f" {topic} " + "-"*20 + "\n")
                file.write(text)

    def summarize_file(self, file_path: str, output_path: str) -> None:
        pages = convert_from_path(file_path, 400, thread_count=10)
        processed_pages_info = [self._mark_page_contours(page, index) for index, page in enumerate(pages)]
        doc = self._build_document_dict(processed_pages_info)
        abstract = doc.pop("abstract")

        try:
            del doc['misc']
            del doc['Acknowledgement']
        except Exception:
            pass

        temp_text_list = []
        for topico, text in doc.items():
            if text:
                result = self._split_text(text)

                if len(result) > 1:
                    doc[topico] = " ".join([self._summary_text(text) for text in result])
                else:
                    doc[topico] = self._summary_text(result[0])

                temp_text_list.extend(result)

        final_test = " ".join([self._summary_text(text) for text in self._split_text(" ".join(temp_text_list))])

        with open(f"{output_path}/arquivo.txt", "w") as file:
            file.write("-"*20 + " original abstract " + "-"*20 + "\n")
            file.write(f"{abstract} \n\n\n")
            file.write("-"*20 + " creted abstract " + "-"*20 + "\n")
            file.write(f"{final_test} \n\n\n")
