import cv2
import numpy as np
import imutils
from imutils import contours, perspective
from PIL import Image
import os
from typing import List, Tuple, Optional, Union, Any
from sonatabene.scoretyping import StaffLine, Note, Key

class PParser:
    
    def __init__(self):
        """Initialize the parser with empty attributes."""
        self.original_image = None
        self.image = None
        self.processed_image = None
        self.notes_contours = None
        self.original_shape = None
        self.resized_shape = None
        self.filename = None 
    
    def load_image(self, input_source: Union[str, np.ndarray], filename: Optional[str] = "image.png") -> np.ndarray:
        """
        Load and preprocess an image from either a file path or numpy array.
        
        Args:
            input_source: Either a file path (str) or an image array (np.ndarray)
            filename: Optional filename when input_source is an array
            
        Returns:
            np.ndarray: The preprocessed grayscale image
        """
        if isinstance(input_source, str):
            self.original_image = cv2.imread(input_source)
            self.filename = os.path.basename(input_source)
            if self.original_image is None:
                raise FileNotFoundError(f"Could not load image from path: {input_source}")
        
        elif isinstance(input_source, np.ndarray):
            self.original_image = input_source.copy()
            self.filename = filename
        
        if len(self.original_image.shape) == 3:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = self.original_image.copy()
            
        self.processed_image = cv2.bitwise_not(self.image)
        return self.image
    
    def imwrite(self, path: str, image: np.ndarray, overwrite: bool = False) -> bool:
        """
        Save an image to a file path.
        
        Args:
            path (str): Path where the image will be saved.
            image (numpy.ndarray): The image to save.
            overwrite (bool, optional): Whether to overwrite existing files without prompting. 
                                       Defaults to False.
        
        Returns:
            bool: True if the image was saved successfully, False otherwise.
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if os.path.exists(path):
            if not overwrite:
                user_input = input(f"File {path} already exists. Overwrite? (y/n): ")
                if user_input.lower() != 'y':
                    print("Saving cancelled.")
                    return False
        
        return cv2.imwrite(path, image)
    
    def imshow(self, image: np.ndarray) -> None:
        """
        Display an image in a window until a key is pressed or the window is closed.
        
        Args:
            image (numpy.ndarray): The image to display.
        """
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)
        if key == 27 or cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            
    def resize(self, image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image (numpy.ndarray): The image to resize.
            max_dim (int, optional): Maximum dimension (width or height) in pixels.
            
        Returns:
            numpy.ndarray: The resized image.
            
        Note:
            Updates instance image attributes with resized versions.
        """  
        self.original_shape = image.shape
        if image.shape[0] <= max_dim and image.shape[1] <= max_dim:
            return image
        
        height, width = image.shape[:2]
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
            
        self.resized_shape = (new_width, new_height)
        
        resized = cv2.resize(image, (new_width, new_height))
        
        if image is self.image:
            self.image = resized
            self.processed_image = cv2.bitwise_not(resized)
        elif image is self.processed_image:
            self.processed_image = resized
        
        return resized
    
    def find_contours(self, image: np.ndarray, dilate_iterations: int = 3, 
                      min_contour_area: int = 0, pad_size: int = 0) -> List[np.ndarray]:
        """
        Find contours in an image.
        
        Args:
            image (numpy.ndarray): The input image.
            dilate_iterations (int, optional): Number of dilation iterations to perform.
            min_contour_area (int, optional): Minimum area for a contour to be included. 
            pad_size (int, optional): Padding to add around the image before processing. 
            
        Returns:
            list: List of contours sorted from left to right.
        """
        padded_image = self._add_padding(image, pad_size)
        if len(image.shape) == 3:
            gray_line = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_line = padded_image.copy()
        _, binary_line = cv2.threshold(gray_line, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_image = cv2.dilate(binary_line, kernel, iterations=dilate_iterations)
        cnts = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [c for c in cnts if cv2.contourArea(c) > min_contour_area]
        
        if pad_size:
            for cnt in cnts:
                cnt[:, :, 0] -= pad_size
                cnt[:, :, 1] -= pad_size 
        (cnts, _) = contours.sort_contours(cnts)
        
        return cnts
    
    def find_staff_lines(self, dilate_iterations: int = 3, 
                        min_contour_area: int = 10000, pad_size: int = 0) -> List[StaffLine]:
        """
        Find and return staff lines with their properties.
        
        Returns:
            List[StaffLine]: List of staff lines with their properties and empty note lists.
        """
        staff_line_contours = self.find_contours(self.processed_image, dilate_iterations=dilate_iterations, 
                                      min_contour_area=min_contour_area, pad_size=pad_size)
        
        staff_lines = []
        for index, contour in enumerate(sorted(staff_line_contours, key=lambda c: cv2.boundingRect(c)[1])):
            bounds = cv2.boundingRect(contour)
            staff_lines.append(StaffLine(
                index=index,
                filename=self.filename,
                image=self.extract_element(bounds) ,
                contour=contour,
                bounds=bounds,
                notes=[]
            ))
        
        return staff_lines
    
    def find_notes(self, staff_lines: List[StaffLine], dilate_iterations: int = 2, 
                   min_contour_area: int = 50, pad_size: int = 0,
                   max_horizontal_distance: int = 2, overlap_threshold: float = 0.2) -> List[StaffLine]:
        """
        Find notes for each staff line and return structured data.
        
        Args:
            staff_lines: List of staff lines to process
            dilate_iterations: Number of dilation iterations for contour detection
            min_contour_area: Minimum area for a contour to be considered
            pad_size: Padding around the image
            max_horizontal_distance: Maximum distance for grouping note components
            overlap_threshold: Threshold for considering components as overlapping
        
        Returns:
            List[StaffLine]: List of staff lines with their associated notes.
        """
        self.cleaned_image = self.remove_staff_lines(self.processed_image)
        global_index = 0
        
        for line_index, staff_line in enumerate(staff_lines):

            mask = np.zeros(self.cleaned_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [staff_line.contour], -1, (255), -1)
            
            x, y, w, h = cv2.boundingRect(staff_line.contour)
            
            line_image = cv2.bitwise_and(self.cleaned_image[y:y+h, x:x+w], 
                                       self.cleaned_image[y:y+h, x:x+w], 
                                       mask=mask[y:y+h, x:x+w])
            
            note_contours = self.find_contours(line_image, 
                                             dilate_iterations=dilate_iterations,
                                             min_contour_area=min_contour_area, 
                                             pad_size=pad_size)
            
            note_contours = self.group_note_components(note_contours,
                                                     max_horizontal_distance=max_horizontal_distance,
                                                     overlap_threshold=overlap_threshold)
            
            note_contours = sorted(note_contours, 
                                 key=lambda c: cv2.boundingRect(c)[0])
            
            for relative_index, note_contour in enumerate(note_contours):
                note_bounds = cv2.boundingRect(note_contour)
                relative_pos = (note_bounds[0], note_bounds[1])
                absolute_pos = (x + note_bounds[0], y + note_bounds[1])

                adjusted_contour = note_contour.copy()
                adjusted_contour[:, :, 0] += x
                adjusted_contour[:, :, 1] += y
                
                bounds = (note_bounds[0] + x, note_bounds[1] + y, note_bounds[2], note_bounds[3])
                full_height_bounds = (bounds[0], y, bounds[2], staff_line.bounds[3])
                full_height_image = self.extract_element(full_height_bounds)
                
                note = Note(
                    index=global_index,          
                    relative_index=relative_index,  
                    line_index=line_index,  
                    image=full_height_image,        
                    contour=adjusted_contour,
                    bounds=bounds,
                    full_height_bounds=full_height_bounds,
                    relative_position=relative_pos,
                    absolute_position=absolute_pos
                )
                staff_line.notes.append(note)
                global_index += 1
        
        self.notes_contours = [[note.contour for note in staff.notes] for staff in staff_lines]
        
        return staff_lines
    
    def group_note_components(self, contours: List[np.ndarray], 
                             max_horizontal_distance: int = 10,
                             overlap_threshold: float = 0.8) -> List[np.ndarray]:
        """
        Group contours that likely belong to the same musical note based on horizontal proximity,
        vertical overlap, and containment in a single pass.
        
        Args:
            contours (list): List of contours to group.
            max_horizontal_distance (int): Maximum horizontal distance between 
                                        contours to be considered part of the same group.
            overlap_threshold (float): Minimum overlap ratio to consider a contour as
                                     contained within another.
            
        Returns:
            list: List of merged contours where each group is represented as a single contour.
        """
        if not contours:
            return []

        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        groups = []
        current_group = []
        
        for contour in sorted_contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            
            if not current_group:
                current_group = [(contour, rect)]
                continue
            
            last_x = current_group[-1][1][0] + current_group[-1][1][2]  

            horizontal_dist = x - last_x
            
            if horizontal_dist > max_horizontal_distance:
                if current_group:
                    groups.append(self.__merge_group(current_group))
                current_group = [(contour, rect)]
                continue
            
            should_merge = False
            for group_contour, group_rect in current_group:
                gx, gy, gw, gh = group_rect
                
                vertical_overlap = (y <= gy + gh) and (gy <= y + h)
                
                if vertical_overlap:
                    x_left = max(x, gx)
                    y_top = max(y, gy)
                    x_right = min(x + w, gx + gw)
                    y_bottom = min(y + h, gy + gh)
                    
                    if x_right >= x_left and y_bottom >= y_top:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        area1 = w * h
                        area2 = gw * gh
                        
                        ratio1 = intersection_area / area1 if area1 > 0 else 0
                        ratio2 = intersection_area / area2 if area2 > 0 else 0
                        max_ratio = max(ratio1, ratio2)
                        
                        if max_ratio > overlap_threshold or horizontal_dist <= max_horizontal_distance:
                            should_merge = True
                            break
            
            if should_merge:
                current_group.append((contour, rect))
            else:
                if current_group:
                    groups.append(self.__merge_group(current_group))
                current_group = [(contour, rect)]
        
        if current_group:
            groups.append(self.__merge_group(current_group))
        
        return groups

    def __merge_group(self, group: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Merge a group of contours into a single contour.
        
        Args:
            group: List of tuples containing (contour, bounding_rect)
            
        Returns:
            np.ndarray: Merged contour
        """
        if len(group) == 1:
            return group[0][0]
        
        x_min = min(rect[0] for _, rect in group)
        y_min = min(rect[1] for _, rect in group)
        x_max = max(rect[0] + rect[2] for _, rect in group)
        y_max = max(rect[1] + rect[3] for _, rect in group)
        
        return np.array([
            [[x_min, y_min]],
            [[x_max, y_min]],
            [[x_max, y_max]],
            [[x_min, y_max]]
        ], dtype=np.int32)
    
    def _add_padding(self, image: np.ndarray, pad_size: int = 0) -> np.ndarray:
        """
        Add padding around an image.
        
        Args:
            image (numpy.ndarray): The input image.
            pad_size (int, optional): Size of padding to add on all sides. Defaults to 0.
            
        Returns:
            numpy.ndarray: The padded image.
        """
        if pad_size > 0:
            padded = cv2.copyMakeBorder(
                image,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )
            return padded
        return image
    
    def __mid_point(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

    def draw_contours(self, image: np.ndarray, cnts: List[np.ndarray], 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2, show_midpoints: bool = False) -> np.ndarray:
        """
        Draw contours on an image with bounding boxes, midpoints, and cross lines.
        
        Args:
            original_image (numpy.ndarray): The image to draw on.
            cnts (list): List of contours to draw.
            color (tuple, optional): BGR color for the contour lines.   
            thickness (int, optional): Thickness of the contour lines. 
            show_midpoints (bool, optional): Whether to show midpoints.
            
        Returns:
            numpy.ndarray: Image with drawn contours and annotations.
        """
        orig = image.copy()
        for c in cnts:
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            
            cv2.drawContours(orig, [box.astype("int")], -1, color, thickness)
            
            if show_midpoints:
                tl, tr, br, bl = box
                midpoints = [
                    self.__mid_point(tl, tr),  # top
                    self.__mid_point(bl, br),  # bottom
                    self.__mid_point(tl, bl),  # left
                    self.__mid_point(tr, br)   # right
                ]
                
                for point in midpoints:
                    cv2.circle(orig, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
                
                cv2.line(orig, (int(midpoints[0][0]), int(midpoints[0][1])),
                        (int(midpoints[1][0]), int(midpoints[1][1])), (255, 0, 255), 2)
                cv2.line(orig, (int(midpoints[2][0]), int(midpoints[2][1])),
                        (int(midpoints[3][0]), int(midpoints[3][1])), (255, 0, 255), 2)
        
        return orig
    
    def extract_contours(self, image: np.ndarray, contours: List[np.ndarray], 
                        axis: int = 0, full_height: bool = False) -> List[np.ndarray]:
        """
        Extract regions from an image based on contours, sorted by position.
        
        Args:
            image (numpy.ndarray): The source image.
            contours (list): List of contours to extract.
            axis (int): Sorting axis - 0 for horizontal (left to right), 1 for vertical (top to bottom).
            full_height (bool, optional): Whether to extract the full height of the image for each contour.
            
        Returns:
            list: List of image regions corresponding to each contour.
            
        Raises:
            ValueError: If axis is not 0 or 1.
        """
        if axis == 0:
            sorted_cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[0]) 
        elif axis == 1:
            sorted_cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[1]) 
        else:
            raise ValueError("Axis must be 0 for horizontal or 1 for vertical")
        
        results = []
        img_height = image.shape[0]
        
        for c in sorted_cnts:
            x, y, w, h = cv2.boundingRect(c)
            if full_height:
                region = image[0:img_height, x:x+w]
            else:
                region = image[y:y+h, x:x+w] 
            results.append(region)
        return results
    
    def extract_element(self, bounds : Tuple[int, int, int, int]): 
        x, y, w, h = bounds
        return self.image[y:y+h, x:x+w]
    
    def remove_staff_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Remove horizontal staff lines from a music score image.
        
        Args:
            image (numpy.ndarray): The input image, typically a music score.
            
        Returns:
            numpy.ndarray: Image with staff lines removed.
        """
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
        thresh = cv2.subtract(image, detected_lines)
        return thresh

    def draw_staff_lines(self, image: np.ndarray, staff_lines: List[StaffLine],
                        show_staff_bounds: bool = True,
                        show_staff_contours: bool = True,
                        show_note_bounds: bool = True,
                        show_note_contours: bool = True) -> np.ndarray:
        """
        Draw staff lines and their notes on the image.
        
        Args:
            image: The image to draw on
            staff_lines: List of staff lines objects to draw
            OPTIONAL : 
                show_staff_bounds: Whether to show staff bounding boxes
                show_note_bounds: Whether to show note bounding boxes
                show_staff_contours: Whether to show staff contours
                show_note_contours: Whether to show note contours.
                
        Returns:
            numpy.ndarray: Image with staff lines and notes
        """
        result = self.original_image.copy()
        
        for staff in staff_lines:
            # Staff line
            if show_staff_bounds:
                x, y, w, h = staff.bounds
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green
            
            if show_staff_contours:
                cv2.drawContours(result, [staff.contour], -1, (0, 255, 0), 1)  # Green
            
            # Notes
            for note in staff.notes:
                if show_note_bounds:
                    x, y, w, h = note.bounds
                    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue
                    cv2.putText(result, str(note.index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                if show_note_contours:
                    cv2.drawContours(result, [note.contour], -1, (0, 0, 255), 1)  # Red
        
        return result