import os
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.util import random_noise
import matplotlib.pyplot as plt
import gc
import psutil
from pathlib import Path

DATA_DIR = Path("E:/brain net/data/ROI")
MRI_DIR = Path("E:/brain net/data/wm mri")
FDG_DIR = Path("E:/brain net/data/imcalc FDG")

OUTPUT_DIR_AD_CN = Path("E:/brain net/DATA2/AD_vs_CN")
OUTPUT_DIR_CN_MCI = Path("E:/brain net/DATA2/CN_vs_MCI")
OUTPUT_DIR_AD_MCI = Path("E:/brain net/DATA2/AD_vs_MCI")

for task_dir in [OUTPUT_DIR_AD_CN, OUTPUT_DIR_CN_MCI, OUTPUT_DIR_AD_MCI]:
    (task_dir / "MRI").mkdir(parents=True, exist_ok=True)
    (task_dir / "FDG").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

labels_df = pd.read_csv(DATA_DIR / 'Label.csv')


def rotate_img(img, angle):
    return rotate(img, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)


def flip_img(img, axis):
    return np.flip(img, axis=axis)


def find_brain_bounding_box(image):
    x_min, y_min, x_max, y_max = (image.shape[1], image.shape[2], 0, 0)
    for slice_idx in range(image.shape[0]):
        slice_img = (image[slice_idx] * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(slice_img, (5, 5), 0)
        _, thres = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            curr_x, curr_y, curr_w, curr_h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            x_min = min(x_min, curr_x)
            y_min = min(y_min, curr_y)
            x_max = max(x_max, curr_x + curr_w)
            y_max = max(y_max, curr_y + curr_h)
    x_min = max(x_min - 10, 0)
    y_min = max(y_min - 10, 0)
    x_max = min(x_max + 10, image.shape[1])
    y_max = min(y_max + 10, image.shape[2])
    return x_min, y_min, x_max, y_max


def visualize_bounding_box(image, x_min, y_min, x_max, y_max, slice_idx=25, save_path=None):
    import matplotlib.patches as patches
    slice_img = image[slice_idx]
    fig, ax = plt.subplots(1)
    ax.imshow(slice_img, cmap='gray')
    rect = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def standardize(volume, mean, std):
    volume = volume.astype(np.float32)
    if std == 0:
        return volume - mean
    return (volume - mean) / std


def resize_to_input_shape(img, mean, std):
    target_shape = (50, 128, 128)
    x_min, y_min, x_max, y_max = find_brain_bounding_box(img)
    cropped = img[:, x_min:x_max, y_min:y_max]
    resized = resize(cropped, target_shape, mode='constant', anti_aliasing=True)
    resized = standardize(resized, mean, std)
    return resized


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1).reshape(shape)
    return distored_image


def gamma_correction(image, gamma):
    image = np.clip(image, 0, 1)
    invGamma = 1.0 / gamma
    return np.clip(np.power(image, invGamma), 0, 1)


def adjust_contrast(image, contrast_factor):
    mean = np.mean(image)
    return np.clip((image - mean) * contrast_factor + mean, 0, 1)


def augment_image(img):
    augmented = img.copy()
    if len(augmented.shape) == 2:
        augmented = np.expand_dims(augmented, axis=0)

    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        augmented = rotate_img(augmented, angle)

    if np.random.rand() > 0.5:
        alpha = np.random.uniform(1, 3)
        sigma = np.random.uniform(0.5, 1.0)
        augmented = elastic_transform(augmented, alpha, sigma)

    if np.random.rand() > 0.5:
        gamma_value = np.random.uniform(0.8, 1.2)
        augmented = gamma_correction(augmented, gamma_value)

    if np.random.rand() > 0.5:
        contrast_factor = np.random.uniform(0.8, 1.2)
        augmented = adjust_contrast(augmented, contrast_factor)

    if np.random.rand() > 0.5:
        augmented = random_noise(augmented, mode='gaussian', mean=0, var=0.0001, clip=True)

    if np.random.rand() > 0.5:
        sigma = np.random.uniform(0.5, 1.0)
        augmented = gaussian_filter(augmented, sigma=sigma)

    augmented = np.clip(augmented, 0, 1)
    return augmented


def process_image(image_path, mean, std, visualize=False):
    img = nib.load(str(image_path)).get_fdata().astype(np.float32)
    middle = img.shape[2] // 2
    start = max(0, middle - 25)
    end = min(img.shape[2], middle + 25)
    img = img[:, :, start:end]
    if img.shape[2] < 50:
        padding_size = 50 - img.shape[2]
        padding = img[:, :, -(padding_size + 1):-1][:, :, ::-1]
        img = np.concatenate([img, padding], axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = resize_to_input_shape(img, mean, std)
    return img


def balance_and_augment_data(X, y):
    unique_labels, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)

    augmented_X = list(X)
    augmented_y = list(y)

    for label, count in zip(unique_labels, counts):
        if count < max_count:
            num_to_add = max_count - count
            class_indices = np.where(y == label)[0]

            for i in range(num_to_add):
                original_index = np.random.choice(class_indices)
                original_image = X[original_index]
                augmented_image = augment_image(original_image)
                augmented_X.append(augmented_image)
                augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)


def print_class_distribution(y, label="Dataset"):
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"{label} class distribution: {class_distribution}")


def compute_dataset_mean_std(data_dir, binary_task=None):
    pixel_count = 0
    mean_sum = 0.0
    sq_diff_sum = 0.0

    for _, row in labels_df.iterrows():
        image_path = list(data_dir.glob(f'**/{row["Subject"]}*.nii'))
        if not image_path:
            continue
        label = row['Group']
        if binary_task == 'AD_vs_CN' and label not in ['AD', 'CN']:
            continue
        if binary_task == 'CN_vs_MCI' and label not in ['CN', 'MCI']:
            continue
        if binary_task == 'AD_vs_MCI' and label not in ['AD', 'MCI']:
            continue

        img = nib.load(str(image_path[0])).get_fdata().astype(np.float32)

        mean_sum += np.sum(img)
        pixel_count += img.size

    global_mean = mean_sum / pixel_count

    for _, row in labels_df.iterrows():
        image_path = list(data_dir.glob(f'**/{row["Subject"]}*.nii'))
        if not image_path:
            continue
        label = row['Group']
        if binary_task == 'AD_vs_CN' and label not in ['AD', 'CN']:
            continue
        if binary_task == 'CN_vs_MCI' and label not in ['CN', 'MCI']:
            continue
        if binary_task == 'AD_vs_MCI' and label not in ['AD', 'MCI']:
            continue

        img = nib.load(str(image_path[0])).get_fdata().astype(np.float32)
        sq_diff_sum += np.sum((img - global_mean) ** 2)

    global_std = np.sqrt(sq_diff_sum / pixel_count)

    print(f"For task {binary_task} on {data_dir.name}, Global Mean: {global_mean}, Global Std: {global_std}")
    return global_mean, global_std


def process_data(data_dir, output_dir, modality, binary_task=None, mean=0, std=1):
    print(f"Processing {modality} images for task {binary_task}...")

    all_samples = []
    for _, row in labels_df.iterrows():
        image_path_list = list(data_dir.glob(f'**/{row["Subject"]}*.nii'))
        if image_path_list:
            image_path = image_path_list[0]
            label = row['Group']
            encoded_label = -1

            if binary_task == 'AD_vs_CN':
                if label == 'AD':
                    encoded_label = 1
                elif label == 'CN':
                    encoded_label = 0
            elif binary_task == 'CN_vs_MCI':
                if label == 'MCI':
                    encoded_label = 1
                elif label == 'CN':
                    encoded_label = 0
            elif binary_task == 'AD_vs_MCI':
                if label == 'AD':
                    encoded_label = 1
                elif label == 'MCI':
                    encoded_label = 0

            if encoded_label != -1:
                img = process_image(image_path, mean, std, visualize=False)
                all_samples.append((img, encoded_label))

    if not all_samples:
        print(f"No samples found for {modality} in task {binary_task}. Skipping.")
        return

    X = np.array([sample[0] for sample in all_samples])
    y = np.array([sample[1] for sample in all_samples])

    print_class_distribution(y, label=f"{modality} Original Dataset")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_SEED
    )

    print_class_distribution(y_train, label=f"{modality} Initial Training Set")
    print_class_distribution(y_val, label=f"{modality} Validation Set")
    print_class_distribution(y_test, label=f"{modality} Test Set")

    print(f"Balancing and augmenting training data for {modality}...")
    X_train_balanced, y_train_balanced = balance_and_augment_data(X_train, y_train)

    print_class_distribution(y_train_balanced, label=f"{modality} Balanced Training Set")

    output_path = output_dir / modality
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving datasets for {modality} to {output_path}")
    np.save(output_path / "X_train.npy", X_train_balanced)
    np.save(output_path / "y_train.npy", y_train_balanced)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "y_val.npy", y_val)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_test.npy", y_test)

    print(f"{modality} processing for task {binary_task} complete.")
    del X, y, X_train_val, y_train_val, X_train, y_train, X_val, y_val, X_test, y_test, X_train_balanced, y_train_balanced
    gc.collect()


def main():
    try:
        print("Starting data preprocessing...")

        mri_mean_adcn, mri_std_adcn = compute_dataset_mean_std(MRI_DIR, binary_task='AD_vs_CN')
        fdg_mean_adcn, fdg_std_adcn = compute_dataset_mean_std(FDG_DIR, binary_task='AD_vs_CN')

        process_data(MRI_DIR, OUTPUT_DIR_AD_CN, "MRI", binary_task='AD_vs_CN', mean=mri_mean_adcn, std=mri_std_adcn)
        process_data(FDG_DIR, OUTPUT_DIR_AD_CN, "FDG", binary_task='AD_vs_CN', mean=fdg_mean_adcn, std=fdg_std_adcn)

        mri_mean_cnmci, mri_std_cnmci = compute_dataset_mean_std(MRI_DIR, binary_task='CN_vs_MCI')
        fdg_mean_cnmci, fdg_std_cnmci = compute_dataset_mean_std(FDG_DIR, binary_task='CN_vs_MCI')

        process_data(MRI_DIR, OUTPUT_DIR_CN_MCI, "MRI", binary_task='CN_vs_MCI', mean=mri_mean_cnmci, std=mri_std_cnmci)
        process_data(FDG_DIR, OUTPUT_DIR_CN_MCI, "FDG", binary_task='CN_vs_MCI', mean=fdg_mean_cnmci, std=fdg_std_cnmci)

        mri_mean_admci, mri_std_admci = compute_dataset_mean_std(MRI_DIR, binary_task='AD_vs_MCI')
        fdg_mean_admci, fdg_std_admci = compute_dataset_mean_std(FDG_DIR, binary_task='AD_vs_MCI')

        process_data(MRI_DIR, OUTPUT_DIR_AD_MCI, "MRI", binary_task='AD_vs_MCI', mean=mri_mean_admci, std=mri_std_admci)
        process_data(FDG_DIR, OUTPUT_DIR_AD_MCI, "FDG", binary_task='AD_vs_MCI', mean=fdg_mean_admci, std=fdg_std_admci)

        print("All data processing is complete.")
        print(f"Final memory usage: {psutil.virtual_memory().percent}%")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()