import cv2
from numpy import ndarray
from pytesseract import image_to_string
from transformers import BartForConditionalGeneration, BartTokenizer


class TextSummarizer:
    _model: BartForConditionalGeneration
    _tokenizer: BartTokenizer

    _MODEL_NAME: str = "facebook/bart-large-cnn"

    def __init__(self) -> None:
        self._model = BartForConditionalGeneration.from_pretrained(self._MODEL_NAME)
        self._tokenizer = BartTokenizer.from_pretrained(self._MODEL_NAME)

    def _extract_contour_text(self, image: ndarray, contour: list[tuple[int, int]]) -> str:
        """Extracts text from a contour.

        Args:
            image (ndarray): The image to extract the text from.
            contour (list[tuple[int, int]]): The contour to extract the text from.

        Returns:
            str: The extracted text.
        """
        croped_img = image[
            contour[0][1]: contour[1][1],
            contour[0][0]: contour[1][0]
        ]

        _, new_image = cv2.threshold(
            src=croped_img,
            thresh=120,
            maxval=255,
            type=cv2.THRESH_BINARY
        )

        return str(image_to_string(image=new_image, config="--psm 6"))

    def _split_text(self, text: str, max_tokens: int = 1000) -> list[str]:
        """_summary_

        Args:
            text (str): _description_
            max_tokens (int, optional): _description_. Defaults to 1000.

        Returns:
            list[str]: _description_
        """
        tokens = self._tokenizer.encode(
            text=text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        if tokens.shape[1] <= max_tokens:
            return [text]

        split_point = len(text)//2

        part1 = text[:split_point]
        part2 = text[split_point:]

        return [part1] + self._split_text(part2)

    def _summary_text(self, input_text: str) -> str:
        """_summary_

        Args:
            input_text (str): _description_

        Returns:
            str: _description_
        """
        input_ids = self._tokenizer.encode(
            text=input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = self._model.generate(
            inputs=input_ids,
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        summary_text = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary_text

    def _preprocess_text(self, text: str) -> str:
        """removes special characters from string

        Args:
            text (str): string to be processed

        Returns:
            str: string with removed special characters
        """
        return text.replace("-\n", "").replace("\n", "").replace("\x0c", "") if text else ""
