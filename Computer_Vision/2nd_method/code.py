import cv2
import numpy as np

def process_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Mask the image to only show the region of interest
    mask = np.zeros_like(edges)
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    # Draw the detected lines on a blank image
    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_image, lines)

    # Overlay the detected lines on the original frame
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return result

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    if lines is None:
        return

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Open the video file or capture the webcam stream
cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Process the current frame
        result = process_image(frame)

        # Display the processed frame
        cv2.imshow('Lane Detection', result)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
