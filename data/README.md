# 📂 Data Directory

This folder is used to store **datasets** used in this repository.

⚠️ **Datasets are NOT included in this repository** due to size and licensing restrictions.

---

## 📥 How to use this folder

1. Download the dataset you want to use (e.g. from Kaggle)
2. Place the files inside this `data/` directory
3. Follow the suggested structure below

---

## 🗂️ Suggested folder structure

You can organize your datasets like this:

```
data/
│
├── raw/                  # Original unmodified datasets
│   └── dataset_name/
│
├── processed/            # Cleaned / transformed data
│   └── dataset_name/
│
├── interim/              # Temporary data generated during preprocessing
│   └── dataset_name/
│
└── external/             # Optional external or third-party data
    └── dataset_name/
```

### Example

```
data/
└── raw/
    └── ct_scans/
        ├── images/
        └── labels.csv
```


---

## 📌 Using the data in your code

To keep things consistent, always reference datasets using **relative paths** from the project root.

Example in Python:

```python
import os

DATA_DIR = "../data/raw/ct_scans"

images_path = os.path.join(DATA_DIR, "images")
labels_path = os.path.join(DATA_DIR, "labels.csv")
```
* 

---

## 🧠 Tip for PyTorch projects

When creating datasets with PyTorch, you can plug this path directly:

```python
dataset = CustomImageDataset(
    annotations_file="data/raw/ct_scans/labels.csv",
    img_dir="data/raw/ct_scans/images"
)
```

---

## 🚫 Git Tracking Notice

All folders suggested in this directory structure are **already listed in `.gitignore`**.

This means:

* Files placed inside `data/` will **not be tracked by Git**
* Your repository will stay lightweight and clean

### ⚠️ Using a custom folder?

If you decide to create a different folder structure inside `data/`, it will be ignored by default, so don't worry!

Keeping datasets out of version control is a **best practice** in Machine Learning projects.

---

## 🔗 Example dataset sources

* Kaggle
* UCI Machine Learning Repository
* Google Dataset Search
* Academic datasets (medical, NLP, etc.)

* ⚠️ All the datasets used in the projects have their links in the beggining of notebook

---

## 💡 Final note

This structure is inspired by common data science project standards and helps keep experiments **organized, reproducible, and scalable**.