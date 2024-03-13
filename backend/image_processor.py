import os

import cv2
import numpy
from numpy import ndarray
from PIL.PpmImagePlugin import PpmImageFile


class ImageProcessor:

    def mark_page_contours(
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
        page_file_location = f"./temp/{page_index}_page.jpg"
        processed_page_location = f"./temp/{page_index}.jpg"

        page.save(page_file_location, "JPEG")
        page_img = orignal_img = cv2.imread(page_file_location)

        page_img = self._preprocess_image(original_img=page_img)
        contours = self._extract_image_contours(original_img=page_img)

        contours_coordinates = [self._write_contours(image=orignal_img, contour=contour) for contour in contours]

        cv2.imwrite(processed_page_location, orignal_img)
        os.remove(os.path.abspath(page_file_location))

        return {"image_path": processed_page_location, "contours_coordinates": contours_coordinates}

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
        image = cv2.cvtColor(src=original_img, code=cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(src=image, ksize=(9, 9), sigmaX=0)
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
                contours[index] = numpy.concatenate((contours[index], contours[index-1]), axis=0)
                contour_to_delete.append(index-1)

        filtered_conturs = [
            contour
            for index, contour in enumerate(contours)
            if index not in contour_to_delete
        ]

        return filtered_conturs

    def _extract_image_contours(self, original_img: ndarray) -> list[ndarray]:
        """Extracts contours from an image.

        Args:
            original_img (ndarray): The original image.

        Returns:
            list[ndarray]: A list of contours found in image.
        """
        contours_found = cv2.findContours(
            image=original_img,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(
            contours_found[0] if len(contours_found) == 2 else contours_found[1],
            key=lambda contour: cv2.boundingRect(contour)[1]
        )

        sorted_contours = self._remove_nearby_contours(
            contours=contours,
            img_hight=original_img.shape[0]
        )

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

        cv2.rectangle(
            img=image,
            pt1=(x_start, y_start),
            pt2=(x_end, y_end),
            color=(255, 0, 255),
            thickness=3
        )

        return [(x_start, y_start), (x_end, y_end)]
