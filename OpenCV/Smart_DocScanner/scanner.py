import cv2
import numpy as np


def order_points(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""

    points = points.reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    # Top-left and bottom-right
    ordered[0] = points[np.argmin(points.sum(axis=1))]
    ordered[2] = points[np.argmax(points.sum(axis=1))]

    # Top-right and bottom-left
    ordered[1] = points[np.argmin(np.diff(points, axis=1))]
    ordered[3] = points[np.argmax(np.diff(points, axis=1))]

    return ordered


# Load image
image = cv2.imread("input/document.jpg")

if image is None:
    print("Could not load image.")
    exit()

# Resize image
image = cv2.resize(image, (800, 600))

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)

# Find document contour
document_contour = None

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 10000:
        continue

    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    if len(approx) == 4:
        document_contour = approx
        break


if document_contour is None:

    print("Document not detected.")

else:

    # Draw detected boundary
    cv2.drawContours(
        image,
        [document_contour],
        -1,
        (0, 255, 0),
        3
    )

    # Order document corners
    points = order_points(document_contour)

    # Calculate output width
    width_top = np.linalg.norm(
        points[1] - points[0]
    )

    width_bottom = np.linalg.norm(
        points[2] - points[3]
    )

    max_width = int(
        max(width_top, width_bottom)
    )

    # Calculate output height
    height_left = np.linalg.norm(
        points[3] - points[0]
    )

    height_right = np.linalg.norm(
        points[2] - points[1]
    )

    max_height = int(
        max(height_left, height_right)
    )

    # Define destination rectangle
    destination = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    # Convert to grayscale
    scanned_gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    # Create scanner-like black and white output
    scanned = cv2.adaptiveThreshold(
        scanned_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Save result
    output_path = "output/scanned_document.jpg"

    success = cv2.imwrite(
        output_path,
        scanned
    )

    if success:
        print(f"Document saved successfully: {output_path}")
    else:
        print("Failed to save the document.")

    # Display results
    cv2.imshow("Detected Document", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("Scanned Document", scanned)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
