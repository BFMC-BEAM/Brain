import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from image_processor_interface import ImageProcessorInterface
from sklearn.linear_model import RANSACRegressor


class LaneDetectionProcessor(ImageProcessorInterface):
    """
    Image processor that detects and draws lane lines.
    """
    
    def __init__(
        self,
        type="simulator"
    ):
        """
        Initializes the lane processor with default or customized settings.
        """
        self.type= type
        self.in_intersection = False
        self.M = None
        self.M_inv = None
        self.previous_left_line = None
        self.previous_right_line = None


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
            UPPER_WHITE = np.array([255, 255, 255])  # Upper bound for white color in HLS

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
        CANNY_LOW_THRESHOLD = 50  # Low threshold for Canny edge detection
        CANNY_HIGH_THRESHOLD = 150  # High threshold for Canny edge detection
        edges = cv2.Canny(
            blurred_image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
        )

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
                thickness=cv2.FILLED  # Fill the rectangle
            )

        # Define the triangular ROI as a numpy array
        triangle = np.array(
            [[
                TRIANGLE_BOTTOM_LEFT,
                TRIANGLE_BOTTOM_RIGHT,
                TRIANGLE_TOP_CENTER,
            ]]
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
            cv2.imshow("ROI Debug", debug_image)

        return masked_image

    def _get_ROI_intersection(self, image):
        """
        Extracts the Region of Interest (ROI) for intersections, dividing the image into left and right regions.

        Args:
            image (np.ndarray): Input image from which the ROI will be extracted.

        Returns:
            tuple: A tuple containing the left and right ROI images.
        """
        # Get the dimensions of the input image
        height, width = image.shape[:2]

        # Define constants for polygon boundaries
        LEFT_POLYGON_TOP = 0  # Top edge of the left polygon
        RIGHT_POLYGON_TOP = height // 3  # Top edge of the right polygon (one-third height)
        CENTER_X = width // 2  # Horizontal center of the image

        # Create empty masks for the left and right regions
        left_mask = np.zeros_like(image)
        right_mask = np.zeros_like(image)

        # Define the polygon for the left region
        lower_left_polygon = np.array([
            [
                (0, height),        # Bottom-left corner
                (0, LEFT_POLYGON_TOP),  # Top-left edge
                (CENTER_X, LEFT_POLYGON_TOP),  # Top-center edge
                (CENTER_X, height)  # Bottom-center edge
            ]
        ], dtype=np.int32)

        # Define the polygon for the right region
        lower_right_polygon = np.array([
            [
                (CENTER_X, height),        # Bottom-center edge
                (CENTER_X, RIGHT_POLYGON_TOP),  # Top-center edge
                (width, RIGHT_POLYGON_TOP),    # Top-right edge
                (width, height)           # Bottom-right corner
            ]
        ], dtype=np.int32)

        # Fill the left polygon in the left mask
        cv2.fillPoly(left_mask, [lower_left_polygon], 255)
        left_image = cv2.bitwise_and(image, left_mask)  # Apply the left mask to the image

        # Fill the right polygon in the right mask
        cv2.fillPoly(right_mask, [lower_right_polygon], 255)
        right_image = cv2.bitwise_and(image, right_mask)  # Apply the right mask to the image

        # Return the masked images for the left and right regions
        return left_image, right_image
     
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
        SIMULATOR_SRC_BOTTOM_LEFT = (50, ysize)  # Bottom-left source point for simulator
        SIMULATOR_SRC_BOTTOM_RIGHT = (550, ysize)  # Bottom-right source point for simulator
        SIMULATOR_SRC_TOP_RIGHT = (550, 180)  # Top-right source point for simulator

        REAL_SRC_TOP_LEFT = (320, 180)  # Top-left source point for real-world
        REAL_SRC_BOTTOM_LEFT = (320, ysize)  # Bottom-left source point for real-world
        REAL_SRC_BOTTOM_RIGHT = (1800, ysize)  # Bottom-right source point for real-world
        REAL_SRC_TOP_RIGHT = (1800, 180)  # Top-right source point for real-world

        # Destination points (common for both cases)
        DST_TOP_LEFT = (PADDING, 0)
        DST_BOTTOM_LEFT = (PADDING, ysize)
        DST_BOTTOM_RIGHT = (xsize - PADDING, ysize)
        DST_TOP_RIGHT = (xsize - PADDING, 0)

        # Define source and destination points based on the environment type
        if self.type == "simulator":
            # Source points for the simulator
            src = np.float32([
                SIMULATOR_SRC_TOP_LEFT,
                SIMULATOR_SRC_BOTTOM_LEFT,
                SIMULATOR_SRC_BOTTOM_RIGHT,
                SIMULATOR_SRC_TOP_RIGHT
            ])
        else:
            # Source points for the real-world
            src = np.float32([
                REAL_SRC_TOP_LEFT,
                REAL_SRC_BOTTOM_LEFT,
                REAL_SRC_BOTTOM_RIGHT,
                REAL_SRC_TOP_RIGHT
            ])

        # Destination points (same for both environments)
        dst = np.float32([
            DST_TOP_LEFT,
            DST_BOTTOM_LEFT,
            DST_BOTTOM_RIGHT,
            DST_TOP_RIGHT
        ])

        # Compute the perspective transform matrix and its inverse
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        # Apply the perspective warp
        warped = cv2.warpPerspective(image, self.M, (xsize, ysize))

        # Convert the warped image to binary using thresholding
        (_, binary_warped) = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

        return binary_warped
   
    def _avg_slope_intercept(self, image, lines):
        """
        Averages and processes the slopes and intercepts of detected lines 
        to produce two representative lane lines (left and right).

        Args:
            image (np.ndarray): Input image used for line detection.
            lines (list): List of detected lines represented by their endpoints.

        Returns:
            tuple: 
                - np.ndarray: Array containing the new left and right lane lines.
                - list: Non-averaged left lines.
                - list: Non-averaged right lines.
        """
        # Lists to store line fits and raw line segments
        left_fit = []
        right_fit = []
        non_avg_left_lines = []
        non_avg_right_lines = []

        # Constants
        MIN_SLOPE_THRESHOLD = 0.2  # Minimum absolute slope to consider a line (to discard horizontal lines)
        INFINITY_SLOPE = float("inf")  # Placeholder for vertical lines

        # Calculate the center of the image
        image_center_x = image.shape[1] // 2

        # Process each detected line
        if lines is not None:
            for line in lines:
                # Extract line endpoints
                x1, y1, x2, y2 = line.reshape(4)

                # Handle vertical lines (to avoid division by zero)
                if x2 - x1 == 0:
                    slope = INFINITY_SLOPE
                    intercept = x1
                else:
                    # Calculate slope and y-intercept using linear regression
                    parameters = np.polyfit((x1, x2), (y1, y2), 1)
                    slope, intercept = parameters

                # Discard nearly horizontal lines based on the slope threshold
                if abs(slope) < MIN_SLOPE_THRESHOLD:
                    continue

                # Determine the position of the line relative to the image center
                mid_x = (x1 + x2) // 2
                if mid_x < image_center_x:  # Line is on the left side
                    non_avg_left_lines.append((x1, y1, x2, y2))
                    left_fit.append((slope, intercept))
                elif mid_x >= image_center_x:  # Line is on the right side
                    non_avg_right_lines.append((x1, y1, x2, y2))
                    right_fit.append((slope, intercept))
                # Segun pendiente, se complica en curvas 
                # if slope < 0:
                #     non_avg_left_lines.append((x1, y1, x2, y2))

                #     left_fit.append((slope, intercept))
                # else:
                #     non_avg_right_lines.append((x1, y1, x2, y2))

                #     right_fit.append((slope, intercept))

        # Generate or retrieve the left line
        if left_fit:
            # Calculate a new average line for the left side
            new_left_line = self._handle_fit(image, left_fit)
            self.previous_left_line = new_left_line
        else:
            # Use the previous line if available, otherwise generate an imaginary line
            if self.previous_left_line is None:
                new_left_line = self._generate_imaginary_line(image, side="left")
            else:
                new_left_line = self.previous_left_line

        # Generate or retrieve the right line
        if right_fit:
            # Calculate a new average line for the right side
            new_right_line = self._handle_fit(image, right_fit)
            self.previous_right_line = new_right_line
        else:
            # Use the previous line if available, otherwise generate an imaginary line
            if self.previous_right_line is None:
                new_right_line = self._generate_imaginary_line(image, side="right")
            else:
                new_right_line = self.previous_right_line

        # Return the averaged lines and the non-averaged line segments
        return np.array([new_left_line, new_right_line]), non_avg_left_lines, non_avg_right_lines

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
        if lines is not None:
            for x1, y1, x2, y2 in lines:
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
        # Declare constants at the beginning
        avg_fit = None
        
        # Calculate the average of the fit lines
        avg_fit = np.mean(fit_lines, axis=0)
            
            # Call a method to generate coordinates
        return self._make_coordinates(image, avg_fit)

    def _generate_imaginary_line(self, image, side):
        """
        Generates an imaginary line on the image.

        Args:
            image (np.ndarray): Input image.
            side (str): Side of the lane ("left" or "right").

        Returns:
            np.ndarray: Coordinates of the imaginary line.
        """
        # Default imaginary line parameters
        LEFT_LINE_SLOPE = -0.65
        RIGHT_LINE_SLOPE = 0.65
        DEFAULT_INTERCEPT_ADJUST = 100  # Default intercept adjustment for imaginary lines

        if side == "left":
    
            intercept = image.shape[0] - LEFT_LINE_SLOPE * DEFAULT_INTERCEPT_ADJUST
        else:
            slope = RIGHT_LINE_SLOPE
            intercept = image.shape[0] - RIGHT_LINE_SLOPE * DEFAULT_INTERCEPT_ADJUST

        return self._make_coordinates(image, (slope, intercept))

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

    def _check_line_angle(self, lines, target_angle, tolerance=10, debug_image=None, print_angle=False):
        """
        Checks if the average line angle is close to the target angle.

        Args:
            lines (list): Detected lines from Hough Transform.
            target_angle (float): Target angle in degrees to check against.
            tolerance (float, optional): Allowable deviation from the target angle. Default is 10 degrees.
            debug_image (np.ndarray, optional): Image for debugging visualization. Default is None.
            print_angle (bool, optional): If True, prints the calculated average angle. Default is False.

        Returns:
            bool: True if the average line angle is within tolerance, otherwise False.
        """
        # Declare constants at the beginning
        DEFAULT_VERTICAL_ANGLE = 90
        DEBUG_LINE_COLOR = (0, 0, 255)  # Red for debug visualization
        TARGET_LINE_COLOR = (0, 255, 0)  # Green for target matching lines
        AVG_LINE_COLOR = (255, 100, 100)  # Blue for average line
        AVG_LINE_LENGTH = 200  # Length of the average line for visualization

        if lines is None or len(lines) == 0:
            return False

        angles = []  # List to store angles of detected lines

        # Iterate over all detected lines
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the angle in degrees
                delta_y = y2 - y1
                delta_x = x2 - x1
                if delta_x == 0:  # Handle vertical lines
                    angle = DEFAULT_VERTICAL_ANGLE
                else:
                    angle = math.degrees(math.atan2(delta_y, delta_x))

                # Draw the line in red for debugging if a debug image is provided
                if debug_image is not None:
                    cv2.line(debug_image, (x1, y1), (x2, y2), DEBUG_LINE_COLOR, 2)

                # If the line angle is within tolerance, add it to the list and draw in green
                if abs(angle - target_angle) <= tolerance:
                    angles.append(angle)
                    if debug_image is not None:
                        cv2.line(debug_image, (x1, y1), (x2, y2), TARGET_LINE_COLOR, 2)

        if not angles:
            return False

        # Calculate the average angle
        avg_angle = np.mean(angles)
        if print_angle:
            print(f"Average Angle: {avg_angle}")

        # Draw the average line for visualization if debug image is provided
        if debug_image is not None:
            h, w = debug_image.shape[:2]
            x_center = w // 2
            y_center = h // 2
            avg_radians = math.radians(avg_angle)

            # Calculate the endpoints of the average line
            x1_avg = int(x_center - AVG_LINE_LENGTH * math.cos(avg_radians))
            y1_avg = int(y_center - AVG_LINE_LENGTH * math.sin(avg_radians))
            x2_avg = int(x_center + AVG_LINE_LENGTH * math.cos(avg_radians))
            y2_avg = int(y_center + AVG_LINE_LENGTH * math.sin(avg_radians))

            cv2.line(debug_image, (x1_avg, y1_avg), (x2_avg, y2_avg), AVG_LINE_COLOR, 4)

        # Check if the average angle is within tolerance
        return abs(avg_angle - target_angle) <= tolerance

    def _check_intersection(self, image):
        """
        Detects intersections in the image by analyzing the angles of lines in specific regions.

        Args:
            image (np.ndarray): Input image for intersection detection.

        Returns:
            None
        """
        # Declare constants for left and right regions
        LEFT_HOUGH_RHO_RESOLUTION = 2  # Rho resolution for the Hough transform (left region)
        LEFT_HOUGH_THETA_RESOLUTION = np.pi / 180  # Theta resolution for the Hough transform (left region)
        LEFT_HOUGH_THRESHOLD = 50  # Threshold for Hough transform (left region)
        LEFT_HOUGH_MIN_LINE_LENGTH = 50  # Minimum line length for the Hough transform (left region)
        LEFT_HOUGH_MAX_LINE_GAP = 30  # Maximum gap allowed between lines (left region)

        RIGHT_HOUGH_RHO_RESOLUTION = 2  # Rho resolution for the Hough transform (right region)
        RIGHT_HOUGH_THETA_RESOLUTION = np.pi / 180  # Theta resolution for the Hough transform (right region)
        RIGHT_HOUGH_THRESHOLD = 150  # Threshold for Hough transform (right region)
        RIGHT_HOUGH_MIN_LINE_LENGTH = 150  # Minimum line length for the Hough transform (right region)
        RIGHT_HOUGH_MAX_LINE_GAP = 50  # Maximum gap allowed between lines (right region)

        # Declare constants for intersection angle checks
        LEFT_ANGLE_RESET = -60  # Target angle for resetting intersection on the left
        RIGHT_ANGLE_RESET = 60  # Target angle for resetting intersection on the right
        LEFT_ANGLE_VALID = 50   # Target angle for detecting an intersection on the left
        RIGHT_ANGLE_VALID = 0   # Target angle for detecting an intersection on the right
        LEFT_TOLERANCE = 10     # Angle tolerance for left lines
        RIGHT_TOLERANCE = 10    # Angle tolerance for right lines
        INTERSECTION_TEXT = "Intersection: YES"
        NO_INTERSECTION_TEXT = "Intersection: NO"

        # Extract regions of interest for left and right sections
        left, right = self._get_ROI_intersection(image)

        # Detect lines using Hough Transform for left and right regions
        left_lines = cv2.HoughLinesP(
            left,
            LEFT_HOUGH_RHO_RESOLUTION,
            LEFT_HOUGH_THETA_RESOLUTION,
            LEFT_HOUGH_THRESHOLD,
            minLineLength=LEFT_HOUGH_MIN_LINE_LENGTH,
            maxLineGap=LEFT_HOUGH_MAX_LINE_GAP,
        )
        right_lines = cv2.HoughLinesP(
            right,
            RIGHT_HOUGH_RHO_RESOLUTION,
            RIGHT_HOUGH_THETA_RESOLUTION,
            RIGHT_HOUGH_THRESHOLD,
            minLineLength=RIGHT_HOUGH_MIN_LINE_LENGTH,
            maxLineGap=RIGHT_HOUGH_MAX_LINE_GAP,
        )

        # Check intersection state based on line angles
        if self.in_intersection:
            left_angle_reset = self._check_line_angle(left_lines, LEFT_ANGLE_RESET, tolerance=LEFT_TOLERANCE)
            right_angle_reset = self._check_line_angle(right_lines, RIGHT_ANGLE_RESET, tolerance=RIGHT_TOLERANCE)
            if left_angle_reset and right_angle_reset:
                self.in_intersection = False  # Reset intersection state
        else:
            left_angle_valid = self._check_line_angle(left_lines, LEFT_ANGLE_VALID, tolerance=40)
            right_angle_valid = self._check_line_angle(right_lines, RIGHT_ANGLE_VALID, tolerance=RIGHT_TOLERANCE)
            self.in_intersection = left_angle_valid and right_angle_valid  # Update intersection state

        # Display intersection status on the output image
        status_text = INTERSECTION_TEXT if self.in_intersection else NO_INTERSECTION_TEXT
        FONT_SCALE = 0.6  # Font scale for displaying text
        FONT_THICKNESS = 1  # Thickness of the font for text
        DIRECTION_COLOR = (0, 255, 0)  # Color for direction text (green)
        INITIAL_X = 10  # Starting X coordinate for the text
        INITIAL_Y = 30  # Starting Y coordinate for the text
        cv2.putText(
            self.output_image,
            status_text,
            (INITIAL_X, INITIAL_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            DIRECTION_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )


    def process_image(self, cv_image):
        """
        Processes the input image to detect and draw lane lines using the Hough transform.
        Also checks for intersections.

        Args:
            cv_image (np.ndarray): Input image.

        Returns:
            np.ndarray: Processed output image with lane lines and intersection status.
        """
        # Hough Transform constants
        HOUGH_RHO_RESOLUTION = 2  # Distance resolution of the Hough transform
        HOUGH_THETA_RESOLUTION = np.pi / 180  # Angular resolution of the Hough transform
        HOUGH_THRESHOLD = 150  # Threshold for detecting lines in the Hough transform
        HOUGH_MIN_LINE_LENGTH = 50  # Minimum line length for Hough line detection
        HOUGH_MAX_LINE_GAP = 150  # Maximum gap between line segments to be connected

        # Line colors
        AVG_LINE_COLOR = (0, 255, 0)  # Color for average lane lines
        NON_AVG_LEFT_COLOR = (0, 0, 255)  # Color for non-average left lane lines
        NON_AVG_RIGHT_COLOR = (255, 0, 0)  # Color for non-average right lane lines

        # Make a copy of the original image
        lane_image = cv_image.copy()

        # Preprocess the image (e.g., thresholding, edge detection)
        border_image = self._preprocessing(lane_image)

        # Get the region of interest (ROI) for the road
        cropped_image = self._get_ROI_road(border_image)

        # Apply perspective warp to get a top-down view
        warped_image = self._warp_image(cropped_image)

        # Detect lines using the Hough transform
        lines = cv2.HoughLinesP(
            warped_image,
            HOUGH_RHO_RESOLUTION,
            HOUGH_THETA_RESOLUTION,
            HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        # Calculate average and non-average (left and right) lane lines
        avg_lines, non_avg_left_lines, non_avg_right_lines = self._avg_slope_intercept(lane_image, lines)

        # If average lane lines were detected, draw them
        if avg_lines is not None:
            # Draw average lane lines
            avg_line_image = self._display_lines(lane_image, avg_lines, AVG_LINE_COLOR)

            # Draw non-average left lane lines
            non_avg_left_line_image = self._display_lines(
                lane_image, non_avg_left_lines, NON_AVG_LEFT_COLOR
            )

            # Draw non-average right lane lines
            non_avg_right_line_image = self._display_lines(
                avg_line_image, non_avg_right_lines, NON_AVG_RIGHT_COLOR
            )

            # Combine all line images
            combined_line_image_a = cv2.addWeighted(
                avg_line_image, 1, non_avg_left_line_image, 1, 0
            )
            combined_line_image_b = cv2.addWeighted(
                combined_line_image_a, 1, non_avg_right_line_image, 1, 0
            )

            # Merge the combined line image with the original image
            self.output_image = cv2.addWeighted(lane_image, 0.8, combined_line_image_b, 1, 0)
        else:
            # If no lines were detected, return the original image
            self.output_image = cv_image

        # Check for intersections using the warped image
        self._check_intersection(warped_image)

        return self.output_image



