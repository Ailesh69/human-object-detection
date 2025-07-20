import cv2
import numpy as np
import argparse

# --- Configuration Constants ---
# Define maximum dimension for display (either width or height will be this max)
# Reduced to 700 to ensure images fit properly on typical laptop screens (1920x1080p)
MAX_DISPLAY_DIMENSION = 700 

# Define consistent colors for bounding boxes (BGR format) - Reverted to original colors
COLOR_FACES = (0, 255, 255)    # Cyan (Original from People1 initial)
COLOR_FACES_SCALED = (0, 255, 0) # Green (Original from People1 scaleFactor)
COLOR_EYES = (0, 0, 255)     # Red (Original for eyes)
COLOR_FULL_BODIES = (0, 255, 0) # Green (Original for fullbody)
COLOR_USER_IMAGE_DETECTIONS = (255, 0, 0) # Blue (Original for user image detections)


# --- Function Definitions ---

def load_and_preprocess_image(image_path):
    """
    Loads an image from the given path, converts it to grayscale,
    and returns both the original color image and the grayscale image.

    Args:
        image_path (str): The relative path to the image file.

    Returns:
        tuple: A tuple containing:
            - cv2.Mat: The original color image (or None if loading fails).
            - cv2.Mat: The grayscale image (or None if loading fails).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the path and file.")
        return None, None

    # Grayscale conversion is often more efficient for cascade detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, gray_image

def resize_image_to_fit_display(image, max_dim=MAX_DISPLAY_DIMENSION):
    """
    Resizes an image to fit within a maximum dimension while maintaining aspect ratio.
    
    Args:
        image (cv2.Mat): The input image.
        max_dim (int): The maximum dimension (width or height) the image should fit within.

    Returns:
        cv2.Mat: The resized image.
    """
    (h, w) = image.shape[:2]
    
    if max(h, w) <= max_dim:
        return image # No resizing needed if already within limits

    if w > h:
        # Landscape or square image, resize based on width
        r = max_dim / float(w)
        dim = (max_dim, int(h * r))
    else:
        # Portrait image, resize based on height
        r = max_dim / float(h)
        dim = (int(w * r), max_dim)

    # Perform the resize
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def detect_and_display_objects(image_color, image_gray, classifier, window_title, detection_params=None, color=(0, 255, 0), thickness=2, wait_key_ms=0):
    """
    Performs object detection using a Haar Cascade classifier, draws bounding boxes,
    and displays the result in a consistently sized window while maintaining aspect ratio.

    Args:
        image_color (cv2.Mat): The original color image on which to draw rectangles.
        image_gray (cv2.Mat): The grayscale image used for detection.
        classifier (cv2.CascadeClassifier): The loaded Haar Cascade classifier.
        window_title (str): The title for the display window.
        detection_params (dict, optional): Dictionary of parameters for detectMultiScale.
                                           Defaults to an empty dictionary if None.
        color (tuple, optional): BGR color tuple for the bounding boxes. Defaults to green (0, 255, 0).
        thickness (int, optional): Thickness of the bounding box line. Defaults to 2.
        wait_key_ms (int, optional): Milliseconds to wait for a key press. 0 means infinite.
                                    Defaults to 0 (wait indefinitely until a key is pressed).
    Returns:
        numpy.ndarray: The array of detected bounding boxes (x, y, w, h).
    """
    if detection_params is None:
        detection_params = {}

    detections = classifier.detectMultiScale(image_gray, **detection_params)

    # Create a copy to draw on, so the original image remains unchanged
    image_with_detections = image_color.copy()

    for (x, y, w, h) in detections:
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), color, thickness)

    # Resize the image to fit the display window while maintaining aspect ratio
    display_image = resize_image_to_fit_display(image_with_detections)

    # Ensure the window is resizable and set its size (using the resized image's dimensions)
    # This ensures the window perfectly fits the aspect-ratio-preserved image
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_image.shape[1], display_image.shape[0])
    
    cv2.imshow(window_title, display_image)
    cv2.waitKey(wait_key_ms)
    cv2.destroyAllWindows()
    
    return detections

# --- Main Program Logic ---

def main():
    """
    Main function to run the human detection project.
    Orchestrates image loading, classifier initialization, and detection processes.
    """
    # Setup command-line argument parsing (still useful for direct testing or specific runs)
    parser = argparse.ArgumentParser(
        description="Human Object Detection using OpenCV Haar Cascades.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--image',
        type=str,
        default='data/people1.jpg', # Default image if none provided
        help='(Optional) Path to an initial image file for demonstration.\n'
             'Example: python your_script_name.py --image data/my_photo.jpg'
    )
    args = parser.parse_args()

    # Initialize Haar Cascade for face detection (used across multiple sections)
    face_detection = cv2.CascadeClassifier("data/frontface.xml")
    print(f"Face cascade classifier loaded: {face_detection is not None}")
    if face_detection.empty():
        print("Error: Face cascade classifier XML not loaded properly. Check path.")
        return 

    # Initialize other classifiers
    eye_detection = cv2.CascadeClassifier("data/eyes.xml")
    print(f"Eye cascade classifier loaded: {eye_detection is not None}")
    if eye_detection.empty():
        print("Error: Eye cascade classifier XML not loaded properly. Check path.")
        eye_detection = None # Set to None so we can check it later

    fullbody_detector = cv2.CascadeClassifier("data/fullbody.xml")
    print(f"Full body cascade classifier loaded: {fullbody_detector is not None}")
    if fullbody_detector.empty():
        print("Error: Full body cascade classifier XML not loaded properly. Check path.")
        fullbody_detector = None # Set to None


    print("\n--- Starting Demo Predictions ---")

    # --- Process the user-provided/default image (initial face detection) ---
    current_image_original, current_image_gray = load_and_preprocess_image(args.image)

    if current_image_original is None:
        print(f"Skipping initial demo image: {args.image}")
    else:
        # The image will be resized for display within detect_and_display_objects
        # No need to pre-resize here for consistent display size.
        # However, for detection, it's often good to work with a consistent size.
        # Let's keep the detection on the original loaded image's grayscale or a reasonable resize.
        # For demo purposes, we'll use the original image's grayscale for detection,
        # and let detect_and_display_objects handle display resizing.

        print(f"\n--- Processing Demo: {args.image} ---")
        print(f"Original image shape: {current_image_original.shape}")
        # Removed resized shape prints as it's now dynamic within display function
        print(f"Grayscale (original size) shape: {current_image_gray.shape}")

        # Initial face detection on the provided image
        initial_face_detections = detect_and_display_objects(
            image_color=current_image_original, # Pass original for aspect ratio resize
            image_gray=current_image_gray,      # Use original grayscale for detection
            classifier=face_detection,
            window_title=f"Demo: Faces - {args.image}",
            color=COLOR_FACES,
            thickness=2,
            wait_key_ms=0
        )
        print(f"Initial face detections on {args.image}: {initial_face_detections}")
        print(f"Number of initial faces detected on {args.image}: {len(initial_face_detections)}")


        # Demonstrate effect of scaleFactor on the provided image
        detections_sf = detect_and_display_objects(
            image_color=current_image_original, # Pass original for aspect ratio resize
            image_gray=current_image_gray,      # Use original grayscale for detection
            classifier=face_detection,
            window_title=f"Demo: Faces (Scale Factor 1.2) - {args.image}",
            detection_params={'scaleFactor': 1.2},
            color=COLOR_FACES_SCALED, # Using the specific color for this demo step
            thickness=3,
            wait_key_ms=0
        )
        print(f"Detections on {args.image} with scaleFactor=1.2: {detections_sf}")


    # --- Continue with other hardcoded demo images (People2.jpg, People3.jpg) ---
    print("\n--- Continuing with additional Demo Images ---")

    # Process People2.jpg (Faces)
    image2_original, image_gray2 = load_and_preprocess_image("data/people2.jpg")
    if image2_original is None:
        print("Skipping people2.jpg due to loading error.")
    else:
        # Detect with minNeighbors to reduce false positives on People2.jpg
        detections2 = detect_and_display_objects(
            image_color=image2_original,
            image_gray=image_gray2,
            classifier=face_detection,
            window_title="Demo: People2 - Faces (minNeighbors 7)",
            detection_params={'scaleFactor': 1.2, 'minNeighbors': 7},
            color=COLOR_FACES,
            thickness=2,
            wait_key_ms=0
        )
        print(f"Detections on People2 with minNeighbors=7: {detections2}")

        # Detect with minSize and maxSize on People2.jpg
        detections2_ms = detect_and_display_objects(
            image_color=image2_original,
            image_gray=image_gray2,
            classifier=face_detection,
            window_title="Demo: People2 - Faces (minSize/maxSize)",
            detection_params={'scaleFactor': 1.2, 'minNeighbors': 7, 'minSize': (20, 20), 'maxSize': (100, 100)},
            color=COLOR_FACES,
            thickness=2,
            wait_key_ms=0
        )
        print(f"Detections on People2 with minSize/maxSize: {detections2_ms}")

    # Eye Detection (on People1.jpg)
    if eye_detection is not None: # Only run if eye classifier loaded successfully
        image3_original, image3_gray = load_and_preprocess_image("data/people1.jpg")
        if image3_original is None:
            print("Skipping eye detection due to image loading error for people1.jpg.")
        else:
            # For combined display, we need to handle resizing manually before drawing
            # This image is now passed to resize_image_to_fit_display directly
            image_to_process = image3_original.copy() # Make a copy to avoid modifying original

            # Perform Face detection on the same image for combined display
            # Use the original grayscale for detection, then resize for display
            face_detections_on_image3 = face_detection.detectMultiScale(image3_gray, scaleFactor = 1.3, minSize = (30,30))

            # Perform Eye detection
            eye_detections = eye_detection.detectMultiScale(image3_gray, scaleFactor = 1.1, minNeighbors=10, maxSize=(60,60))

            # Draw faces first (using consistent color)
            for (x, y, w, h) in face_detections_on_image3:
                cv2.rectangle(image_to_process, (x, y), (x + w, y + h), COLOR_FACES, 2)

            # Draw eyes second (using consistent color)
            for (x, y, w, h) in eye_detections:
                print(f"Eye detection coordinates: {w}, {h}")
                cv2.rectangle(image_to_process, (x, y), (x + w, y + h), COLOR_EYES, 2)

            # Resize the final image with detections to fit the display window
            display_image_combined = resize_image_to_fit_display(image_to_process, max_dim=MAX_DISPLAY_DIMENSION)

            # Ensure the window is resizable and set its size for combined display
            cv2.namedWindow('Demo: People1 - Faces and Eyes Detected', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Demo: People1 - Faces and Eyes Detected', display_image_combined.shape[1], display_image_combined.shape[0])

            cv2.imshow('Demo: People1 - Faces and Eyes Detected', display_image_combined)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
    else:
        print("Skipping eye detection: Classifier not loaded.")

    # Full Body Detection
    if fullbody_detector is not None: # Only run if full body classifier loaded successfully
        fullbody_image_original, gray_fullbody = load_and_preprocess_image("data/people3.jpg")
        if fullbody_image_original is None:
            print("Skipping full body detection due to image loading error for people3.jpg.")
        else:
            # Pass original image to detect_and_display_objects, it will handle resizing
            detection_fullbody = detect_and_display_objects(
                image_color=fullbody_image_original,
                image_gray=gray_fullbody,
                classifier=fullbody_detector,
                window_title="Demo: People3 - Full Body Detected",
                detection_params={'scaleFactor': 1.03, 'minNeighbors': 1},
                color=COLOR_FULL_BODIES,
                thickness=2,
                wait_key_ms=0
            )
    else:
        print("Skipping full body detection: Classifier not loaded.")


    # --- Interactive User Input Loop ---
    print("\n--- Demo Predictions Complete ---")
    while True:
        print("\nWhat would you like to do next?")
        print("1) Enter your own image for detection")
        print("2) Exit the code")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == '1':
            user_image_path = input("Enter the path to your image (e.g., data/my_photo.jpg): ").strip()
            
            user_image_original, user_image_gray = load_and_preprocess_image(user_image_path)

            if user_image_original is None:
                print(f"Could not load image from '{user_image_path}'. Please try again with a valid path.")
                continue # Go back to the menu

            print(f"\n--- Processing User Image: {user_image_path} ---")

            # Perform face detection on user's image
            if face_detection is not None:
                user_face_detections = detect_and_display_objects(
                    image_color=user_image_original, # Pass original for aspect ratio resize
                    image_gray=user_image_gray,      # Use original grayscale for detection
                    classifier=face_detection,
                    window_title=f"User Image: Detected Faces - {user_image_path}",
                    color=COLOR_USER_IMAGE_DETECTIONS,
                    thickness=3,
                    wait_key_ms=0
                )
                print(f"Faces detected on {user_image_path}: {len(user_face_detections)}")
                if len(user_face_detections) == 0:
                    print(f"No faces detected in '{user_image_path}'. This may be due to image quality, lighting, or subject pose, which can affect Haar Cascade performance.")
                    print("Try an image with a clear, frontal face, or one of the demo images like 'data/people1.jpg'.")
            else:
                print("Face detection skipped: Classifier not loaded.")

            # --- Optional: Add Eye/Full Body detection for user image here ---
            if eye_detection is not None:
                user_eye_detections = detect_and_display_objects(
                    image_color=user_image_original, # Pass original for aspect ratio resize
                    image_gray=user_image_gray,      # Use original grayscale for detection
                    classifier=eye_detection,
                    window_title=f"User Image: Detected Eyes - {user_image_path}",
                    color=COLOR_EYES,
                    thickness=2,
                    wait_key_ms=0
                )
                print(f"Eyes detected on {user_image_path}: {len(user_eye_detections)}")

            if fullbody_detector is not None:
                user_fullbody_detections = detect_and_display_objects(
                    image_color=user_image_original, # Pass original for aspect ratio resize
                    image_gray=user_image_gray,      # Use original grayscale for detection
                    classifier=fullbody_detector,
                    window_title=f"User Image: Detected Full Bodies - {user_image_path}",
                    color=COLOR_FULL_BODIES,
                    thickness=2,
                    wait_key_ms=0
                )
                print(f"Full Bodies detected on {user_image_path}: {len(user_fullbody_detections)}")


        elif choice == '2':
            print("Exiting the program. Goodbye!")
            break # Exit the loop and end the program
        else:
            print("Invalid choice. Please enter '1' or '2'.")

# --- Entry Point ---
if __name__ == "__main__":
    main()
