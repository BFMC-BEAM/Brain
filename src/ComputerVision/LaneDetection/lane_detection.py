import cv2
import numpy as np
import math
from src.ComputerVision.LaneDetection.image_processor_interface import ImageProcessorInterface

class LaneDetectionProcessor(ImageProcessorInterface):
    """
    Image processor that detects and draws lane lines.
    """

    def __init__(self, type="simulator"):
        """
        Initializes the lane processor with default or customized settings.
        """
        self.type = type
        self.in_intersection = False
        self.M = None
        self.M_inv = None
        self.previous_left_line = None
        self.previous_right_line = None
        self.in_curve = False
        self.in_cross_walk = False
        self.in_intersection = False
        self.deviation = 0
        self.direction = None

    def _adjust_parameters(self, gray_image):
        """
        Adjusts the parameters for edge detection based on the lighting conditions 
        of the input image using its histogram and intensity analysis.

        Args:
            gray_image (np.ndarray): Grayscale version of the input image.

        Returns:
            tuple: A pair of integers representing the low and high thresholds for Canny edge detection.
        """
        
        # Constants for light condition thresholds
        LOW_LIGHT_MEAN_THRESHOLD = 60  # Mean intensity threshold for low light conditions
        NORMAL_LIGHT_MEAN_MIN = 60     # Minimum mean intensity for normal light conditions
        NORMAL_LIGHT_MEAN_MAX = 150    # Maximum mean intensity for normal light conditions
        HIGH_LIGHT_THRESHOLD = 0.5     # Percentage threshold for bright pixels to indicate high light
        LOW_LIGHT_THRESHOLD = 0.5      # Percentage threshold for dark pixels to indicate low light

        # Constants for Canny edge detection thresholds under different light conditions
        CANNY_LOW_LIGHT = (20, 80)     # (low_threshold, high_threshold) for low light conditions
        CANNY_NORMAL_LIGHT = (50, 150) # (low_threshold, high_threshold) for normal light conditions
        CANNY_HIGH_LIGHT = (70, 200)   # (low_threshold, high_threshold) for high light conditions
        CANNY_MIXED_LIGHT = (40, 120)  # Default values for mixed or unclear light conditions

        # Calculate the histogram of the grayscale image
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        # Compute the total number of pixels in the image
        total_pixels = np.sum(histogram)

        # Determine the proportion of low and high intensity pixels
        low_intensity_pixels = np.sum(histogram[:50]) / total_pixels  # Proportion of dark pixels
        high_intensity_pixels = np.sum(histogram[200:]) / total_pixels  # Proportion of bright pixels

        # Calculate the mean intensity of the image
        mean_intensity = np.mean(gray_image)
        # print("intensidad media:", mean_intensity)
        # Adjust the Canny thresholds based on light conditions
        if mean_intensity < LOW_LIGHT_MEAN_THRESHOLD or low_intensity_pixels > LOW_LIGHT_THRESHOLD:
            # Low light conditions
            canny_low_threshold, canny_high_threshold = CANNY_LOW_LIGHT
        elif NORMAL_LIGHT_MEAN_MIN <= mean_intensity <= NORMAL_LIGHT_MEAN_MAX \
            and low_intensity_pixels < 0.3 and high_intensity_pixels < 0.3:
            # Normal light conditions with balanced intensities
            canny_low_threshold, canny_high_threshold = CANNY_NORMAL_LIGHT
        elif high_intensity_pixels > HIGH_LIGHT_THRESHOLD:
            # High light conditions
            canny_low_threshold, canny_high_threshold = CANNY_HIGH_LIGHT
        else:
            # Mixed or unclear light conditions
            canny_low_threshold, canny_high_threshold = CANNY_MIXED_LIGHT

        return canny_low_threshold, canny_high_threshold

    def _preprocessing(self, image):
        """
        Preprocess the input image to extract edges for further processing.

        Args:
            image (np.ndarray): Input image to preprocess.

        Returns:
            np.ndarray: Edge-detected image.
        """

        if self.type == "simulator":
            # Convert to grayscale for simulator images
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Convert to HLS color space for real-world images
            hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

            # Lower and upper bound for white color in HLS
            LOWER_WHITE = np.array([0, 160, 10])  # Lower bound for white color in HLS
            UPPER_WHITE = np.array(
                [255, 255, 255]
            )  # Upper bound for white color in HLS

            # Create a mask for white colors in the HLS space
            mask = cv2.inRange(hls, LOWER_WHITE, UPPER_WHITE)

            # Apply the mask to isolate white areas
            hls_result = cv2.bitwise_and(image, image, mask=mask)

            # Convert the masked result to grayscale
            gray_image = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        THRESHOLD_BINARY = 165  # Threshold value for binary conversion
        THRESHOLD_MAX_VALUE = 255  # Maximum value for binary thresholding
        _, binary_image = cv2.threshold(
            gray_image, THRESHOLD_BINARY, THRESHOLD_MAX_VALUE, cv2.THRESH_BINARY
        )

        # Apply Gaussian blur to reduce noise
        GAUSSIAN_BLUR_KERNEL = (5, 5)  # Kernel size for Gaussian blur
        blurred_image = cv2.GaussianBlur(binary_image, GAUSSIAN_BLUR_KERNEL, 0)

        # Perform Canny edge detection
        CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD = self._adjust_parameters(gray_image)


        CANNY_LOW_THRESHOLD = 50  # Low threshold for Canny edge detection
        CANNY_HIGH_THRESHOLD = 150  # High threshold for Canny edge detection
        edges = cv2.Canny(blurred_image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

        return edges

    def _get_ROI_road(self, image, show=False):
        """
        Extracts the Region of Interest (ROI) for the road from an image.

        Args:
            image (np.ndarray): Input image from which the ROI will be extracted.
            show (bool): Whether to display the debug visualization of the ROI.

        Returns:
            np.ndarray: Image with the ROI applied.
        """
        # Get the height and width of the input image
        height, width = image.shape[:2]

        # Initialize a mask with the same dimensions as the input image
        mask = np.zeros_like(image)

        # Default coordinates for the triangular ROI in real-world conditions
        TRIANGLE_BOTTOM_LEFT = (320, height)
        TRIANGLE_BOTTOM_RIGHT = (1800, height)
        TRIANGLE_TOP_CENTER = (1012, 301)

        # Adjust coordinates if working in a simulator environment
        if self.type == "simulator":
            # Simulator-specific coordinates for the triangular ROI
            TRIANGLE_BOTTOM_LEFT = (50, 480)
            TRIANGLE_BOTTOM_RIGHT = (550, 480)
            TRIANGLE_TOP_CENTER = (331, 95)

            # Define the rectangle area in the simulator (covers the bottom part of the image)
            RECTANGLE_BOTTOM_LEFT = (0, 301)
            RECTANGLE_TOP_RIGHT = (width, height)

            # Add the rectangular area to the mask
            cv2.rectangle(
                mask,
                RECTANGLE_BOTTOM_LEFT,
                RECTANGLE_TOP_RIGHT,
                255,  # White color for the mask
                thickness=cv2.FILLED,  # Fill the rectangle
            )

        # Define the triangular ROI as a numpy array
        triangle = np.array(
            [
                [
                    TRIANGLE_BOTTOM_LEFT,
                    TRIANGLE_BOTTOM_RIGHT,
                    TRIANGLE_TOP_CENTER,
                ]
            ]
        )

        # Add the triangular area to the mask
        cv2.fillPoly(mask, triangle, 255)

        # Apply the mask to the input image using a bitwise AND operation
        masked_image = cv2.bitwise_and(image, mask)

        # Debug visualization (optional, if `show` is True)
        if show:
            # Create a copy of the original image for debugging
            debug_image = image.copy()

            # Overlay the triangular ROI in green on the debug image
            cv2.fillPoly(debug_image, triangle, (0, 255, 0))

            # Blend the original and debug images with some transparency
            alpha = 0.5  # Transparency factor
            debug_image = cv2.addWeighted(image, 1 - alpha, debug_image, alpha, 0)

            # Display the debug visualization
            return debug_image
        
        return masked_image

    def _warp_image(self, image):
        """
        Applies a perspective warp to the input image.

        Args:
            image (np.ndarray): Input image to be warped.

        Returns:
            np.ndarray: Binary image after perspective transformation.
        """
        # Get the dimensions of the input image
        ysize, xsize = image.shape[:2]

        # Constants for padding and source/destination points
        PADDING = 0  # Padding for the destination points
        SIMULATOR_SRC_TOP_LEFT = (50, 180)  # Top-left source point for simulator
        SIMULATOR_SRC_BOTTOM_LEFT = (
            50,
            ysize,
        )  # Bottom-left source point for simulator
        SIMULATOR_SRC_BOTTOM_RIGHT = (
            550,
            ysize,
        )  # Bottom-right source point for simulator
        SIMULATOR_SRC_TOP_RIGHT = (550, 180)  # Top-right source point for simulator

        REAL_SRC_TOP_LEFT = (320, 180)  # Top-left source point for real-world
        REAL_SRC_BOTTOM_LEFT = (320, ysize)  # Bottom-left source point for real-world
        REAL_SRC_BOTTOM_RIGHT = (
            1800,
            ysize,
        )  # Bottom-right source point for real-world
        REAL_SRC_TOP_RIGHT = (1800, 180)  # Top-right source point for real-world

        # Destination points (common for both cases)
        DST_TOP_LEFT = (PADDING, 0)
        DST_BOTTOM_LEFT = (PADDING, ysize)
        DST_BOTTOM_RIGHT = (xsize - PADDING, ysize)
        DST_TOP_RIGHT = (xsize - PADDING, 0)

        # Define source and destination points based on the environment type
        if self.type == "simulator":
            # Source points for the simulator
            src = np.float32(
                [
                    SIMULATOR_SRC_TOP_LEFT,
                    SIMULATOR_SRC_BOTTOM_LEFT,
                    SIMULATOR_SRC_BOTTOM_RIGHT,
                    SIMULATOR_SRC_TOP_RIGHT,
                ]
            )
        else:
            # Source points for the real-world
            src = np.float32(
                [
                    REAL_SRC_TOP_LEFT,
                    REAL_SRC_BOTTOM_LEFT,
                    REAL_SRC_BOTTOM_RIGHT,
                    REAL_SRC_TOP_RIGHT,
                ]
            )

        # Destination points (same for both environments)
        dst = np.float32(
            [DST_TOP_LEFT, DST_BOTTOM_LEFT, DST_BOTTOM_RIGHT, DST_TOP_RIGHT]
        )

        # Compute the perspective transform matrix and its inverse
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        # Apply the perspective warp
        warped = cv2.warpPerspective(image, self.M, (xsize, ysize))

        # Convert the warped image to binary using thresholding
        (_, binary_warped) = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

        return binary_warped

    def _filter_outliers(self, fits, lines, image, window_name=None):
        """
        Filters out line fits that are outliers based only on the intercept.

        Args:
            fits (list): List of tuples containing (slope, intercept).
            lines (list): List of original lines corresponding to fits.
            image (np.ndarray): Original image used for visualization.
            window_name (str): Name of the window to display filtered lines.
            threshold (float): Number of standard deviations to use for filtering.

        Returns:
            list: Filtered list of fits.
        """
        intercepts = np.array([fit[1] for fit in fits])

        intercept_mean = np.mean(intercepts)

        # Filter fits based on the intercept threshold
        filtered_fits = []
        filtered_lines = []
        for fit, line in zip(fits, lines):
            if abs(fit[1] - intercept_mean) <= 200 or len(filtered_fits) == 0:
                filtered_fits.append(fit)
            else:
                filtered_lines.append(line)

        # Display filtered lines on the image
        if filtered_lines and window_name is not None:
            filtered_image = image.copy()
            for line in filtered_lines:
                x1, y1, x2, y2 = line

                # Unwarp the line coordinates if necessary
                if hasattr(self, "M_inv") and self.M_inv is not None:
                    pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    pts = cv2.perspectiveTransform(pts[None, :, :], self.M_inv)[0]
                    x1, y1 = int(pts[0][0]), int(pts[0][1])
                    x2, y2 = int(pts[1][0]), int(pts[1][1])

                # Draw the unwarped line
                cv2.line(filtered_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow(window_name, filtered_image)
            cv2.waitKey(1)

        return filtered_fits

    def _line_classifier(self, image, lines):
        """
        Processes detected lines to average and classify them into left and right lane lines.

        Args:
            image (np.ndarray): Input image where lines were detected.
            lines (list): List of detected lines, each represented by their endpoints (x1, y1, x2, y2).

        Returns:
            tuple:
                - np.ndarray: Array containing the new left and right lane lines.
                - list: Raw (non-averaged) left lines.
                - list: Raw (non-averaged) right lines.
                - list: Mid lines potentially indicating crosswalks or other features.
        """
        # Lists for storing slopes and intercepts for left and right lines
        left_fit = []
        right_fit = []

        # Variables to store the final averaged lines
        new_left_line = np.array([None], dtype=object)
        new_right_line = np.array([None], dtype=object)

        # Lists for storing raw line segments before averaging
        non_avg_left_lines = []
        non_avg_right_lines = []
        mid_lines = []
        horizontal_lines = []
        # Constants for filtering lines
        MIN_SLOPE_THRESHOLD = (
            0.2  # Minimum slope magnitude to filter out nearly horizontal lines
        )
        INFINITY_SLOPE = float("inf")  # Placeholder for vertical lines (infinite slope)

        # Constants for detecting crosswalk lines
        CROSSWALK_X1_MIN = 150  # Minimum x1 coordinate for a crosswalk
        CROSSWALK_X1_MAX = 450  # Maximum x1 coordinate for a crosswalk
        CROSSWALK_LENGTH_MAX = 100  # Maximum length (x1 - x2) for a crosswalk line

        # Check if any lines are provided
        if lines is not None:
            for line in lines:
                # Extract line coordinates
                x1, y1, x2, y2 = line.reshape(4)

                # Handle vertical lines (avoid division by zero)
                if x2 - x1 == 0:
                    slope = INFINITY_SLOPE
                    intercept = x1
                else:
                    # Calculate slope and intercept of the line using linear regression
                    parameters = np.polyfit((x1, x2), (y1, y2), 1)
                    slope, intercept = parameters

                # Discard nearly horizontal lines based on the slope threshold
                if abs(slope) < MIN_SLOPE_THRESHOLD:
                    horizontal_lines.append((x1, y1, x2, y2))
                    continue

                # Detect potential crosswalks based on line position and length
                if (
                    CROSSWALK_X1_MIN < x1 < CROSSWALK_X1_MAX
                    and abs(x1 - x2) < CROSSWALK_LENGTH_MAX
                ):
                    mid_lines.append((x1, y1, x2, y2))
                    continue

                # Classify the line as left or right based on its slope
                if slope < 0:  # Negative slope indicates a left line
                    non_avg_left_lines.append((x1, y1, x2, y2))
                    left_fit.append((slope, intercept))
                else:  # Positive slope indicates a right line
                    non_avg_right_lines.append((x1, y1, x2, y2))
                    right_fit.append((slope, intercept))

        # Process left lines to generate the final averaged left lane line
        if len(left_fit) != 0:
            left_fit = self._filter_outliers(left_fit, non_avg_left_lines, image)
            new_left_line = self._handle_fit(image, left_fit)
            self.previous_left_line = new_left_line

        # Process right lines to generate the final averaged right lane line
        if len(right_fit) != 0:
            right_fit = self._filter_outliers(right_fit, non_avg_right_lines, image)
            new_right_line = self._handle_fit(image, right_fit)
            self.previous_right_line = new_right_line

        # Return the final lane lines and raw line classifications
        return (
            np.array([new_left_line, new_right_line], dtype=object),
            non_avg_left_lines,
            non_avg_right_lines,
            mid_lines,
            horizontal_lines,
        )

    def _display_lines(self, image, lines, color):
        """
        Draws lines on an image. If a perspective transformation exists,
        the lines are transformed back to the original perspective.

        Args:
            image (np.ndarray): Input image where lines will be displayed.
            lines (list): List of lines to be drawn, each represented by endpoints (x1, y1, x2, y2).
            color (tuple): Color of the lines to be drawn (BGR format).

        Returns:
            np.ndarray: Image with the lines drawn on it.
        """
        # Constants
        LINE_THICKNESS = 2  # Thickness of the lines to be drawn

        # Initialize a blank image with the same dimensions as the input image
        line_image = np.zeros_like(image)

        # Draw each line on the blank image

        for line in lines:
            if len(line) == 4:
                x1, y1, x2, y2 = line
                # If a perspective transformation exists, transform the line back
                if self.M_inv is not None:
                    pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    # Apply the inverse perspective transformation
                    pts = cv2.perspectiveTransform(pts[None, :, :], self.M_inv)[0]
                    x1, y1 = int(pts[0][0]), int(pts[0][1])
                    x2, y2 = int(pts[1][0]), int(pts[1][1])

                # Draw the line on the blank image
                cv2.line(line_image, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        return line_image

    def _handle_fit(self, image, fit_lines):
        """
        Handles the computation of the average fit line and generates coordinates.

        Args:
            image (np.ndarray): Input image.
            fit_lines (list of np.ndarray): List of lines to calculate the average.

        Returns:
            np.ndarray or None: Coordinates for the average fit line, or None if an error occurs.
        """
        # Constants
        avg_fit = None

        # Calculate the average of the fit lines
        avg_fit = np.mean(fit_lines, axis=0)

        return self._make_coordinates(image, avg_fit)

    def _make_coordinates(self, image, line_parameters):
        """
        Calculates the coordinates of a line based on its slope and intercept.

        Args:
            image (np.ndarray): Input image.
            line_parameters (tuple): Slope and intercept of the line.

        Returns:
            np.ndarray: Array containing the coordinates [x1, y1, x2, y2].
        """

        # Constants
        slope, intercept = line_parameters
        Y2_SCALE = 2 / 5  # Scale for the height of lines in the region of interest
        y1 = image.shape[0]  # Bottom of the image
        y2 = int(y1 * Y2_SCALE)  # A scaled y-coordinate for the line's top point

        if slope == float("inf"):
            # Vertical line case
            x1 = intercept
            x2 = intercept
        else:
            # General case: calculate x-coordinates based on the slope and intercept
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])

    def _check_line_angle(self, lines, target_angle, tolerance):
        """
        Checks if the average line angle is close to the target angle.

        Args:
            lines (list): Detected lines from Hough Transform.
            target_angle (float): Target angle in degrees to check against.
            tolerance (float, optional): Allowable deviation from the target angle. Default is 10 degrees.

        Returns:
            bool: True if the average line angle is within tolerance, otherwise False.
        """
        # Constants
        DEFAULT_VERTICAL_ANGLE = 90  # Angle for vertical lines
        TOLERANCE_THRESHOLD = tolerance  # Maximum allowable deviation from target angle

        # Return False if no lines are provided
        if lines is None or len(lines) == 0:
            return False

        # List to store the angles of detected lines
        angles = []

        # Iterate over all detected lines
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the differences in x and y coordinates
                delta_y = y2 - y1
                delta_x = x2 - x1

                # Handle vertical lines (to avoid division by zero)
                if delta_x == 0:
                    angle = DEFAULT_VERTICAL_ANGLE
                else:
                    # Calculate the angle in degrees using arctangent
                    angle = math.degrees(math.atan2(delta_y, delta_x))
                # Check if the line angle is within the specified tolerance
                if abs(angle - target_angle) <= TOLERANCE_THRESHOLD:
                    angles.append(angle)

        # Return False if no angles fall within the tolerance
        if not angles:
            return False

        # Calculate the average angle of the filtered lines
        avg_angle = np.mean(angles)

        # Return True if the average angle is within tolerance, otherwise False
        return abs(avg_angle - target_angle) <= TOLERANCE_THRESHOLD

    def _check_curve(self, left_line, right_line):
        """
        Determines if the path is a left curve, right curve, or straight,
        based on the angles of the left and right lane lines.

        Args:
            left_line (list): Detected left line from Hough Transform.
            right_line (list): Detected right line from Hough Transform.

        Returns:
            None
        """
        # Constants for curve detection
        LEFT_CURVE_TARGET_ANGLE = -50  # Expected angle for left line in a right curve
        RIGHT_CURVE_TARGET_ANGLE = -135  # Expected angle for right line in a left curve
        LEFT_CURVE_TOLERANCE = (
            10  # Allowed deviation from the target angle for left line
        )
        RIGHT_CURVE_TOLERANCE = (
            10  # Allowed deviation from the target angle for right line
        )

        # Constants for text annotation
        TEXT_POSITION = (0, 15)  # Position where the text will be drawn on the image
        FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX  # Font type for the text
        FONT_SCALE = 0.6  # Font scale for the text
        FONT_THICKNESS = 1  # Thickness of the text
        TEXT_COLOR = (0, 255, 0)  # Green color for the text (BGR format)

        # Variables to determine the curve type
        is_left_curve = False
        is_right_curve = False

        # Check if the left and right lines are present
        left_line_present = len(left_line) == 4
        right_line_present = len(right_line) == 4

        # Evaluate the left line for a right curve
        if left_line_present:
            # Format left line as expected by _check_line_angle method
            left_line = [[[left_line[0], left_line[1], left_line[2], left_line[3]]]]
            is_right_curve = self._check_line_angle(
                left_line, LEFT_CURVE_TARGET_ANGLE, LEFT_CURVE_TOLERANCE
            )

        # Evaluate the right line for a left curve
        if right_line_present:
            # Format right line as expected by _check_line_angle method
            right_line = [
                [[right_line[0], right_line[1], right_line[2], right_line[3]]]
            ]
            is_left_curve = self._check_line_angle(
                right_line, RIGHT_CURVE_TARGET_ANGLE, RIGHT_CURVE_TOLERANCE
            )

        # Determine the type of path based on the curve detections
        if is_left_curve:
            path_text = "Left curve"
        elif is_right_curve:
            path_text = "Right curve"
        else:
            path_text = "Straight"

        # Annotate the path type on the output image
        cv2.putText(
            self.output_image,
            path_text,
            TEXT_POSITION,  # Position for text
            FONT_TYPE,  # Font type
            FONT_SCALE,  # Font scale
            TEXT_COLOR,  # Text color
            FONT_THICKNESS,  # Text thickness
            cv2.LINE_AA,  # Line type for better rendering
        )

    def _check_intersection(self, lines):
        """
        Checks if an intersection is detected based on the number of horizontal lines and crosswalk status.

        Args:
            lines (list): List of horizontal lines to check for intersection.

        Returns:
            None
        """
        # Constants for text annotation
        TEXT_POSITION = (0, 40)  # Position where the text will be drawn on the image
        FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX  # Font type for the text
        FONT_SCALE = 0.6  # Font scale for the text
        FONT_THICKNESS = 1  # Thickness of the text
        TEXT_COLOR = (0, 255, 0)  # Green color for the text (BGR format)

        # Constants for intersection detection
        MIN_LINES_FOR_INTERSECTION = (
            2  # Minimum number of lines to detect an intersection
        )
        MAX_LINES_FOR_INTERSECTION = (
            5  # Maximum number of lines to detect an intersection
        )

        # Initial text indicating whether a crosswalk is detected
        text = "Intersection: "
        # Determine if an intersection is detected
        if (
            MIN_LINES_FOR_INTERSECTION <= len(lines) <= MAX_LINES_FOR_INTERSECTION
            and self.in_cross_walk is False
        ):
            text += "yes"  # Intersection detected
        else:
            text += "no"  # No intersection detected

        # Annotate the intersection detection result on the output image
        cv2.putText(
            self.output_image,
            text,
            TEXT_POSITION,  # Position for text
            FONT_TYPE,  # Font type
            FONT_SCALE,  # Font scale
            TEXT_COLOR,  # Text color
            FONT_THICKNESS,  # Text thickness
            cv2.LINE_AA,  # Line type for better rendering
        )

    def _check_cross_walk(self, lines):
        """
        Determines if a crosswalk is detected based on the number of mid-lines.

        Args:
            lines (list): List of mid-lines potentially indicating a crosswalk.

        Returns:
            None
        """
        # Constants for text annotation
        MIN_LINES_FOR_CROSSWALK = 3  # Minimum number of lines to detect a crosswalk
        TEXT_POSITION = (0, 65)  # Position for the text annotation on the image
        FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX  # Font type for the annotation
        FONT_SCALE = 0.6  # Scale of the font
        FONT_THICKNESS = 1  # Thickness of the font
        TEXT_COLOR = (0, 255, 0)  # Color of the text (green in BGR format)

        # Initial text indicating whether a crosswalk is detected
        text = "Crosswalk: "
        if len(lines) < MIN_LINES_FOR_CROSSWALK:
            self.in_cross_walk = False
            text += "no"
        else:
            self.in_cross_walk = True
            text += "yes"

        # Annotate the crosswalk detection result on the output image
        cv2.putText(
            self.output_image,
            text,
            TEXT_POSITION,  # Position for the text
            FONT_TYPE,  # Font type
            FONT_SCALE,  # Font scale
            TEXT_COLOR,  # Text color
            FONT_THICKNESS,  # Text thickness
            cv2.LINE_AA,  # Line type for better rendering
        )

    def _check_deviation(self, left_line, right_line):
        """
        Calculates the central point of the lane based on the detected lines and deviation from center.
        Args:
            left_line (np.ndarray): Coordinates of the left line [x1, y1, x2, y2] (optional).
            right_line (np.ndarray): Coordinates of the right line [x1, y1, x2, y2] (optional).
        Returns:
            tuple: Coordinates (x, y) of the central point or None if insufficient data.
        """
        # Early exit if no lines are detected
        if len(left_line) == 1 and len(right_line) == 1:
            return None
        # print("tengo:",left_line, right_line)
        # Constants for calculations
        LANE_WIDTH_METERS = 3.5  # Estimated lane width in meters
        PIXELS_PER_METER = 200  # Approximation of pixels per meter in the image
        OFFSET_PIXELS = int(LANE_WIDTH_METERS * PIXELS_PER_METER / 2)  # Half lane width in pixels
        CIRCLE_RADIUS = 5  # Radius of the circle to mark the center point
        CIRCLE_COLOR = (255, 0, 255)  # Color of the circle (magenta in BGR format)
        TEXT_COLOR = (255, 255, 255)  # Color of the text (white in BGR format)
        TEXT_SCALE = 0.5  # Font scale for text annotation
        TEXT_THICKNESS = 1  # Font thickness for text annotation
        LINE_TYPE = cv2.LINE_AA  # Anti-aliased line type for text
        # Calculate image center in the x-axis
        image_center_x = self.output_image.shape[1] // 2
        center_x, center_y = 0, 0
        # Case 1: Both left and right lines are detected
        if len(left_line) != 1 and len(right_line) != 1:
            # Calculate the bottom midpoint of both lines
            left_bottom_x = (left_line[0] + left_line[2]) // 2
            left_bottom_y = (left_line[1] + left_line[3]) // 2
            right_bottom_x = (right_line[0] + right_line[2]) // 2
            right_bottom_y = (right_line[1] + right_line[3]) // 2
            # Average the midpoints to find the center
            center_x = (left_bottom_x + right_bottom_x) // 2
            center_y = (left_bottom_y + right_bottom_y) // 2
        # Case 2: Only the left line is detected (right curve scenario)
        elif len(left_line) != 1 and len(right_line) == 1:
            x1, y1, x2, y2 = left_line
            center_x = ((x1 + x2) // 2) + OFFSET_PIXELS
            center_y = (y1 + y2) // 2
        # Case 3: Only the right line is detected (left curve scenario)
        elif len(left_line) == 1 and len(right_line) != 1:
            x1, y1, x2, y2 = right_line
            center_x = ((x1 + x2) // 2) - OFFSET_PIXELS
            center_y = (y1 + y2) // 2
        # else:
        # Calculate deviation from the center of the image in meters
        deviation = (center_x - image_center_x) / PIXELS_PER_METER
        # Determine the direction of deviation
        if deviation > 0:
            direction = "right"
        elif deviation < 0:
            direction = "left"
        else:
            direction = "straight"
        # Mark the center point on the image
        cv2.circle(self.output_image, (center_x, center_y), CIRCLE_RADIUS, CIRCLE_COLOR, -1)
        # Display deviation and direction information
        deviation_text = f"Deviation: {abs(deviation):.2f}cm {direction}"
        cv2.putText(
            self.output_image,
            deviation_text,
            (center_x + 10, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            LINE_TYPE,
        )
        return deviation, direction
    
    def _check_lane(self, lines, lane_image, display=True):
        # Line colors for visualization
        AVG_LINE_COLOR = (0, 255, 0)  # Green for averaged lane lines
        NON_AVG_LEFT_COLOR = (0, 0, 255)  # Red for non-averaged left lane lines
        NON_AVG_RIGHT_COLOR = (255, 0, 0)  # Blue for non-averaged right lane lines
        # Initialize line variables
        (
            avg_lines,
            non_avg_left_lines,
            non_avg_right_lines,
            mid_lines,
            horizontal_lines,
            new_left_line,
            new_right_line
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )
        # Classify lines into average and non-average left/right lanes
        (
            avg_lines,
            non_avg_left_lines,
            non_avg_right_lines,
            mid_lines,
            horizontal_lines,
        ) = self._line_classifier(lane_image, lines)
        if avg_lines[0] is not None:
            new_left_line = avg_lines[0]
        if avg_lines[1] is not None:
            new_right_line = avg_lines[1]
        # Draw averaged lane lines
        avg_line_image = self._display_lines(lane_image, avg_lines, AVG_LINE_COLOR)
        if display == True:
            # Draw non-averaged left lane lines
            non_avg_left_line_image = self._display_lines(
                lane_image, non_avg_left_lines, NON_AVG_LEFT_COLOR
            )
            # Draw non-averaged right lane lines
            non_avg_right_line_image = self._display_lines(
                avg_line_image, non_avg_right_lines, NON_AVG_RIGHT_COLOR
            )
            # Combine all drawn lines into one image
            combined_line_image_a = cv2.addWeighted(
                avg_line_image, 1, non_avg_left_line_image, 1, 0
            )
            combined_line_image_b = cv2.addWeighted(
                combined_line_image_a, 1, non_avg_right_line_image, 1, 0
            )
            # Merge the combined line image with the original image
            self.output_image = cv2.addWeighted(
                lane_image, 0.8, combined_line_image_b, 1, 0
            )
        return new_left_line, new_right_line, mid_lines, horizontal_lines
    
    def process_image(self, cv_image):
        """
        Processes the input image to detect and draw lane lines using the Hough transform.
        Also checks for intersections, curves, and crosswalks.

        Args:
            cv_image (np.ndarray): Input image.

        Returns:
            np.ndarray: Processed output image with lane lines and intersection status.
        """
        # Constants for Hough Transform parameters
        HOUGH_RHO_RESOLUTION = (
            2  # Distance resolution of the Hough transform (in pixels)
        )
        HOUGH_THETA_RESOLUTION = (
            np.pi / 180
        )  # Angular resolution of the Hough transform (in radians)
        HOUGH_THRESHOLD = (
            150  # Minimum number of intersections in Hough space to consider a line
        )
        HOUGH_MIN_LINE_LENGTH = 50  # Minimum length of a line segment to be considered
        HOUGH_MAX_LINE_GAP = 150  # Maximum gap between line segments to connect them

        # Line colors for visualization
        AVG_LINE_COLOR = (0, 255, 0)  # Green for averaged lane lines
        NON_AVG_LEFT_COLOR = (0, 0, 255)  # Red for non-averaged left lane lines
        NON_AVG_RIGHT_COLOR = (255, 0, 0)  # Blue for non-averaged right lane lines

        # Preprocess the input image
        lane_image = cv_image.copy()
        border_image = self._preprocessing(
            lane_image
        )  # Apply preprocessing (e.g., edge detection)
        # Extract the region of interest (ROI) for the road
        cropped_image = self._get_ROI_road(border_image)

        # Apply perspective warp to get a bird's-eye view
        warped_image = self._warp_image(cropped_image)
        # cv2.imshow("Warped Image", warped_image)  # Debug visualization

        # Detect lines using the Hough Transform
        lines = cv2.HoughLinesP(
            warped_image,
            HOUGH_RHO_RESOLUTION,
            HOUGH_THETA_RESOLUTION,
            HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        # Initialize line variables
        (
            avg_lines,
            non_avg_left_lines,
            non_avg_right_lines,
            mid_lines,
            horizontal_lines
        ) = (
            None,
            None,
            None,
            None,
            None
        )
        new_left_line, new_right_line = None, None
        #print (lines)
        if lines is not None:
            new_left_line, new_right_line, mid_lines, horizontal_lines = self._check_lane(lines, lane_image, True)
            ret = self._check_deviation(new_left_line, new_right_line)
            if ret is not None:
                self.deviation, self.direction = ret
            self._check_curve(new_left_line, new_right_line)
            self._check_cross_walk(mid_lines)
            self._check_intersection(horizontal_lines)
        else:
            # If no lines are detected, retain the original image
            self.output_image = cv_image


        return self.output_image 

    def get_parameters(self):
        return self.deviation, self.direction