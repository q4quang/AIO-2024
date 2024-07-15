# Bài tập 1: Numpy Cơ Bản
# Câu hỏi 1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

arr = np.arange(0, 10, 1)
print("Result 1st: ", arr)

# Câu hỏi 2
arr = np.ones((3, 3)) > 0
# arr = np.ones((3 ,3), dtype = bool)
# arr = np.full((3 ,3), fill_value = True, dtype = bool)
print("Result 2nd: ", arr)

# Câu hỏi 3
arr = np.arange(0, 10)
print("Result 3rd: ", arr[arr % 2 == 1])

# Câu hỏi 4
arr[arr % 2 == 1] = -1
print("Result 4th: ", arr)

# Câu hỏi 5
arr = np.arange(10)
arr_2d = arr.reshape(2, -1)
print("Result 5th: ", arr_2d)

# Câu hỏi 6
arr1 = np.arange(10).reshape(2, -1)
arr2 = np.repeat(1, 10).reshape(2, -1)
c = np.concatenate([arr1, arr2], axis=0)
print("Result 6th: C = \n", c)

# Câu hỏi 7
d = np.concatenate([arr1, arr2], axis=1)
print("Result 7th: D = \n", d)

# Câu hỏi 8
arr = np.array([1, 2, 3])
print("Result 8th:", np.repeat(arr, 3))
print(np.tile(arr, 3))

# Câu hỏi 9
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.nonzero((a >= 5) & (a <= 10))
print("Result 9th:", a[index])

# Câu hỏi 10


def maxx(x, y):
    if x >= y:
        return x
    else:
        return y


a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

pair_max = np.vectorize(maxx, otypes=[float])
print("Result 10th:", pair_max(a, b))

# Câu hỏi 11
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
print("Result 11th:", np.where(a < b, b, a))

# Bài tập 2: Xử lý ảnh
# Câu hỏi 12, 13, 14


def convert_to_grayscale(img: np.ndarray, method: str) -> np.ndarray:
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must have 3 color channels (RGB).")

    img = img.copy()
    method = method.lower()

    if method == 'lightness':
        return 0.5 * np.max(img, axis=-1) + 0.5 * np.min(img, axis=-1)
    elif method == 'average':
        # 0.33 * img[..., 0] + 0.33*img[..., 1] + 0.34*img[..., 2]
        return np.mean(img, axis=-1)
    elif method == 'luminosity':
        luminosity_values = [0.21, 0.72, 0.07]
        return np.dot(img, luminosity_values)
    else:
        raise ValueError(
            "Invalid method. Choose from 'lightness', 'average', or 'luminosity'.")


def plot_images(orig_img: np.ndarray, gray_img: np.ndarray, method: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title(f'Grayscale Image ({method})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


img = mpimg.imread('Module_02/dog.jpeg')

method = 'Lightness'
img_gray = convert_to_grayscale(img, method=method)
print(img_gray[0, 0])
plot_images(img, img_gray, method=method)

method = 'Average'
img_gray = convert_to_grayscale(img, method=method)
print(img_gray[0, 0])
plot_images(img, img_gray, method=method)

method = 'Luminosity'
img_gray = convert_to_grayscale(img, method=method)
print(img_gray[0, 0])
plot_images(img, img_gray, method=method) 

# Bài tập 3: Dữ liệu dạng bảng
# Câu hỏi 15

df = pd.read_csv('Module_02/advertising.csv')
df.head()

data = df.to_numpy()

print("Max:", np.max(data[:, 3]))
print(np.argmax(data[:, 3]))

# Câu hỏi 16
print("TV mean:", np.mean(data[:, 0]))

# Câu hỏi 17
print("Sales >= 20:", np.sum(data[:, 3] >= 20))

# Câu hỏi 18
sales_mask = data[:, 3] >= 15
print("18th", np.mean(data[sales_mask, 1]))

# Câu hỏi 19
newspaper_col = data[:, 2]
mean_newspaper = np.mean(newspaper_col)
print("19th", np.sum(data[newspaper_col > mean_newspaper, 3]))

# Câu hỏi 20


def get_score(value, mean) -> str:
    if value > mean:
        return 'Good'
    if value < mean:
        return 'Bad'
    return 'Average'


mean_sales = np.mean(data[:, 3])
scores = np.array([get_score(value, mean_sales) for value in data[:, 3]])
scores[7:10]

scores = np.where(
    data[:, 3] > mean_sales,
    'Good',
    np.where(
        data[:, 3] < mean_sales,
        'Bad',
        'Average'
    )
)

print("20th", scores[7:10])

# Câu hỏi 21
nearest_sales = np.abs(data[:, 3] - mean_sales).argmin()
nearest_sales = data[nearest_sales, 3]

scores = np.array([get_score(value, nearest_sales) for value in data[:, 3]])
print("21st", scores[7:10])
