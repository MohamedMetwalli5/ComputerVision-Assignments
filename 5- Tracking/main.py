import cv2
import glob
import numpy as np
from skimage.filters import sobel_h, sobel_v


def track(template, img, corner_coords, W, tolerance=1e-3):
    img_h, img_w = template.shape[:2]
    y_min, x_min, y_max, x_max = corner_coords

    y_coords, x_coords = np.meshgrid(
        np.arange(y_min, y_max+1), np.arange(x_min, x_max+1))

    y_coords = y_coords.ravel()
    x_coords = x_coords.ravel()

    ones = np.ones_like(y_coords)
    zeros = np.zeros_like(y_coords)

    jacobian_template_row = np.vstack(
        [x_coords, zeros, y_coords, zeros, ones, zeros]).T
    jacobian_img_row = np.vstack(
        [zeros, x_coords, zeros, y_coords, zeros, ones]).T
    J = np.hstack([jacobian_template_row, jacobian_img_row]).reshape(-1, 2, 6)

    grad_y = sobel_h(img)
    grad_x = sobel_v(img)

    patch_template = template[y_coords, x_coords]

    for _ in range(50):
        img_grad_y = cv2.warpAffine(
            grad_y, W[:2, :], img.shape[::-1])[y_coords, x_coords]
        img_grad_x = cv2.warpAffine(
            grad_x, W[:2, :], img.shape[::-1])[y_coords, x_coords]
        patch_img = cv2.warpAffine(
            img, W[:2, :], img.shape[::-1])[y_coords, x_coords]
        cv2.imshow('warp', cv2.rectangle(cv2.warpAffine(
            img, W[:2, :], img.shape[::-1]), (x_min, y_min),
            (x_max, y_max), (0, 255, 0), 2))

        grad_mat = np.vstack([img_grad_x, img_grad_y]).T

        grad_times_W = np.einsum('ijk,ikm->ijm', grad_mat[:, None, :], J)
        H = np.einsum('ikj,ikm->ijm', grad_times_W, grad_times_W).sum(axis=0)

        precomputed_prod = np.linalg.inv(H)

        b = (patch_template - patch_img).reshape(-1, 1, 1)

        A_times_b = (grad_times_W * b).sum(axis=0)
        delta_p = precomputed_prod @ A_times_b.T

        W[0, 0] += delta_p[0]
        W[1, 0] += delta_p[1]

        W[0, 1] += delta_p[2]
        W[1, 1] += delta_p[3]

        W[0, 2] += delta_p[4]
        W[1, 2] += delta_p[5]

        if (delta_p**2).mean() <= tolerance:
            break

    corners = np.array([
        [x_min, x_max, x_max, x_min],
        [y_min, y_min, y_max, y_max],
        [1, 1, 1, 1]
    ])

    corners = np.linalg.inv(W) @ corners
    corners /= corners[-1, :]
    corners = np.round(corners).astype(np.int32)

    min_coords = corners.min(axis=1)
    max_coords = corners.max(axis=1)

    new_y_min, new_x_min, new_y_max, new_x_max = min_coords[
        1], min_coords[0], max_coords[1], max_coords[0]

    return W, new_y_min, new_x_min, new_y_max, new_x_max


'''
def track(template, img, corner_coords, W=np.eye(3), tolerance=1e-3):
    img_h, img_w = template.shape[:2]
    y_min, x_min, y_max, x_max = corner_coords

    y_coords, x_coords = np.meshgrid(
        np.arange(y_min, y_max+1), np.arange(x_min, x_max))
    # y_coords, x_coords = np.meshgrid(
    #     np.arange(img_h), np.arange(img_w))

    y_coords = y_coords.ravel()
    x_coords = x_coords.ravel()

    ones = np.ones_like(y_coords)
    zeros = np.zeros_like(y_coords)

    jacobian_template_row = np.vstack(
        [y_coords, zeros, x_coords, zeros, ones, zeros]).T
    jacobian_img_row = np.vstack(
        [zeros, y_coords, zeros, x_coords, zeros, ones]).T
    J = np.hstack([jacobian_template_row, jacobian_img_row]).reshape(-1, 2, 6)

    # grad_y = sobel_h(img)
    # grad_x = sobel_v(img)

    # grad_mat = np.vstack([img_grad_y, img_grad_x]).T

    # grad_times_W = np.einsum('ijk,ikm->ijm', grad_mat[:, None, :], J)
    # H = np.einsum('ikj,ikm->ijm', grad_times_W, grad_times_W).sum(axis=0)

    # patch_template = template[y_coords, x_coords]

    # precomputed_prod = np.linalg.inv(H)
    patch_template = template[y_coords, x_coords]
    coords_mat = np.vstack([y_coords, x_coords, ones])

    while True:
        new_coords_mat = W @ coords_mat
        new_coords_mat = new_coords_mat / new_coords_mat[-1, :]

        # new_y_coords = np.round(new_coords_mat[0, :]).astype(np.int32)
        # np.clip(new_y_coords, 0, img_h-1, out=new_y_coords)
        # new_x_coords = np.round(new_coords_mat[1, :]).astype(np.int32)
        # np.clip(new_x_coords, 0, img_w-1, out=new_x_coords)

        # jacobian_template_row = np.vstack(
        #     [new_y_coords, zeros, new_x_coords, zeros, ones, zeros]).T
        # jacobian_img_row = np.vstack(
        #     [zeros, new_y_coords, zeros, new_x_coords, zeros, ones]).T
        # J = np.hstack([jacobian_template_row, jacobian_img_row]
        #               ).reshape(-1, 2, 6)

        img_grad_y = cv2.warpAffine(sobel_h(img), W[:2, :], img.shape[::-1])[
            y_coords, x_coords]
        img_grad_x = cv2.warpAffine(sobel_v(img), W[:2, :], img.shape[::-1])[
            y_coords, x_coords]

        img = cv2.warpAffine(img, W[:2, :], img.shape[::-1])

        grad_mat = np.vstack([img_grad_y, img_grad_x]).T

        grad_times_W = np.einsum('ijk,ikm->ijm', grad_mat[:, None, :], J)
        H = np.einsum('ikj,ikm->ijm', grad_times_W, grad_times_W).sum(axis=0)

        precomputed_prod = np.linalg.inv(H)

        patch_img = img[y_coords, x_coords]

        b = (patch_template - patch_img).reshape(-1, 1, 1)

        A_times_b = (grad_times_W * b).sum(axis=0)
        delta_p = precomputed_prod @ A_times_b.T

        W[0, 0] += delta_p[0]
        W[1, 0] += delta_p[1]

        W[0, 1] += delta_p[2]
        W[1, 1] += delta_p[3]

        W[0, 2] += delta_p[4]
        W[1, 2] += delta_p[5]

        if (delta_p**2).mean() <= tolerance:
            break

    corners = np.array([
        [y_min, y_min, y_max, y_max],
        [x_min, x_max, x_max, x_min],
        [1, 1, 1, 1]
    ])

    corners = np.linalg.inv(W) @ corners
    corners /= corners[-1, :]
    corners = np.round(corners).astype(np.int32)

    min_coords = corners.min(axis=1)
    max_coords = corners.max(axis=1)

    new_y_min, new_x_min, new_y_max, new_x_max = min_coords[
        0], min_coords[1], max_coords[0], max_coords[1]

    return W, new_y_min, new_x_min, new_y_max, new_x_max
'''

img_paths = sorted(
    glob.glob('./tracking_data/car/*'))
template = cv2.imread(img_paths[0], 0) / 255.0
y_min, x_min, y_max, x_max = 95, 115, 285, 345
W = np.eye(3)

for path in img_paths:
    img = cv2.imread(path, 0) / 255.0
    W, y1, x1, y2, x2 = track(
        template, img, (y_min, x_min, y_max, x_max), W, 0.07)

    bbox_img = cv2.rectangle(img.copy(), (x1, y1),
                             (x2, y2), (0, 255, 0), 2)

    cv2.imshow('tracked', bbox_img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
