import cv2
import glob
import numpy as np
import os


def extract_template_data(template, bbox_coords):
    y_min, x_min, y_max, x_max = bbox_coords

    y_coords, x_coords = np.meshgrid(
        np.arange(y_min, y_max), np.arange(x_min, x_max))

    y_coords = y_coords.ravel()
    x_coords = x_coords.ravel()

    points = np.vstack([x_coords, y_coords, np.ones_like(x_coords)])
    ones = np.ones_like(y_coords)
    zeros = np.zeros_like(y_coords)

    patch_template = template[y_coords, x_coords]

    jacobian_first_row = np.vstack(
        [x_coords, zeros, y_coords, zeros, ones, zeros]).T
    jacobian_second_row = np.vstack(
        [zeros, x_coords, zeros, y_coords, zeros, ones]).T
    jacobian = np.hstack([jacobian_first_row, jacobian_second_row]
                         ).reshape(-1, 2, 6)

    corners = np.array([
        [x_min, x_max],
        [y_min, y_max],
        [1, 1]
    ])

    return patch_template, jacobian, points, corners


def track(patch_template, img, J, points, corners, W, tolerance=1e-3, max_iters=100):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    for _ in range(max_iters):
        warped_points = (W @ points).astype(np.int32)
        warped_x_coords = warped_points[0, :]
        warped_y_coords = warped_points[1, :]

        img_grad_y = grad_y[warped_y_coords, warped_x_coords]
        img_grad_x = grad_x[warped_y_coords, warped_x_coords]
        patch_img = img[warped_y_coords, warped_x_coords]

        grad_mat = np.vstack([img_grad_x, img_grad_y]).T

        grad_times_W = np.einsum('ijk,ikm->ijm', grad_mat[:, None, :], J)
        H = np.einsum('ikj,ikm->ijm', grad_times_W, grad_times_W).sum(axis=0)

        H_inv = np.linalg.pinv(H)

        b = (patch_template - patch_img).reshape(-1, 1)

        A_times_b = (grad_times_W.reshape(-1, 6).T @ b)
        delta_p = (H_inv @ A_times_b).ravel()
        W[0] += delta_p[::2]
        W[1] += delta_p[1::2]

        if np.linalg.norm(delta_p) <= tolerance:
            break

    warped_corners = W @ corners
    warped_corners = np.round(warped_corners).astype(np.int32)

    return W, warped_corners[:2].T


name = 'car'    # ['car', 'landing']
box_coords = {
    'car': [95, 115, 285, 345],
    'landing': [78, 435, 140, 565]
}

max_iters = {
    'car': 80,
    'landing': 150
}

tolerances = {
    'car': 1e-4,
    'landing': 1e-5
}

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

img_paths = sorted(
    glob.glob(os.getcwd() + f'/tracking_data/{name}/*'))
template = clahe.apply(cv2.imread(img_paths[0], 0)).astype(np.float32)

videoWriter = cv2.VideoWriter(
    f'output/{name}.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, template.shape[::-1])

max_iter = max_iters[name]
tolerance = tolerances[name]

W = np.eye(3, dtype=np.float64)

patch_template, J, points, corners = extract_template_data(
    template, box_coords[name])

for path in img_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_img = clahe.apply(gray).astype(np.float32)
    W, warped_corners = track(
        patch_template, equalized_img, J, points, corners, W, tolerance, max_iter)

    bbox_img = cv2.rectangle(
        img.copy(), warped_corners[0], warped_corners[1], (0, 255, 0), 2)

    videoWriter.write(bbox_img)
    cv2.imshow('tracked', bbox_img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videoWriter.release()
