import cv2
import numpy as np
import os

class PanaromaStitcher:
    def __init__(self):
        self.homography_matrix_list = []  # Initialize a list to hold homography matrices

    def stitch_images(self, img1, img2):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Use FLANN-based matcher to find matches
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Find matches
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Store good matches using Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Check if we have enough good matches
        if len(good_matches) < 4:
            print("Not enough matches found.")
            return None, None  # Return None if not enough matches

        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

        if H is None:
            print("Homography could not be computed.")
            return None, None

        # Store the homography matrix
        self.homography_matrix_list.append(H)

        # Get dimensions from images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Calculate the dimensions for the stitched image
        pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)

        combined_width = int(max(w1, dst[2][0][0]))
        combined_height = int(max(h1, dst[2][0][1]))

        # Create a new blank image
        stitched_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Place the first image in the stitched image
        stitched_image[0:h1, 0:w1] = img1

        # Warp the second image
        warped_img2 = cv2.warpPerspective(img2, H, (combined_width, combined_height))

        # Blend images to reduce seams
        mask1 = (stitched_image > 0).astype(np.uint8)
        mask2 = (warped_img2 > 0).astype(np.uint8)

        # Blend images
        stitched_image = stitched_image * mask1 + warped_img2 * mask2 * (1 - mask1)
        stitched_image = np.clip(stitched_image, 0, 255).astype(np.uint8)

        # Cropping: Convert to grayscale for thresholding
        gray_stitched = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_stitched, 1, 255, cv2.THRESH_BINARY)

        # Find contours and crop the stitched image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            stitched_image = stitched_image[y:y + h, x:x + w]

        return stitched_image, H

    def resize_image(self, img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        return cv2.resize(img, (width, height))

    def make_panaroma_for_images_in(self, path):
        images = []
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = self.resize_image(img, 50)  # Resize the image to 50% of its original size
                    images.append(img)
                else:
                    print(f"Error loading image: {img_path}")

        if len(images) < 2:
            raise ValueError("Not enough images to stitch.")

        temp_stitched = images[0]
        stitched_image = temp_stitched.copy()  # Create a separate panorama to hold the entire result

        for i in range(1, len(images)):
            print(f'Stitching image {i + 1}/{len(images)}...')

            # Check if temp_stitched is None before slicing
            if temp_stitched is None:
                print("Previous stitching resulted in None. Skipping to next image.")
                continue

            # Only take the rightmost part of the current stitched image
            rightmost_part = temp_stitched[:, temp_stitched.shape[1] - int(temp_stitched.shape[1] * 0.3):]

            # Stitch using the rightmost part
            temp_stitched, H = self.stitch_images(rightmost_part, images[i])

            if temp_stitched is None:
                print(f"Skipping stitching with image {i + 1} due to insufficient matches.")
                continue  # Skip this iteration if stitching failed

            # Combine the new stitch with the panorama
            panorama_update, H = self.stitch_images(stitched_image, temp_stitched)

            if panorama_update is None:
                print(f"Skipping panorama update with image {i + 1} due to insufficient matches.")
                continue  # Skip if panorama update failed

            stitched_image = panorama_update

        return stitched_image, self.homography_matrix_list
