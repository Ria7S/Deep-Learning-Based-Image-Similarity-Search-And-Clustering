# Deep-Learning-Based-Image-Similarity-Search-And-Clustering

A deep learning–based project that performs **content-based image retrieval**. It uses a **pre-trained VGG16 CNN model** to extract deep features from images, applies **PCA for dimensionality reduction**, and uses **cosine similarity** to retrieve the most similar images to a query image.

---

## 🎯 Objective

The goal is to build a system that can **identify and retrieve images with similar visual content** from a dataset—similar to how Google Images or Pinterest visual search functions.

---

## ⚙️ Workflow Overview

1. **Dataset Download**

   * Uses the [Cat Dataset from Kaggle](https://www.kaggle.com/datasets/crawford/cat-dataset) (CC0 license).
   * Images are organized into folders for training and testing.

2. **Pre-trained Model Loading**

   * Loads **VGG16** from `keras.applications`, pretrained on **ImageNet**.
   * Utilizes the model’s **fully connected layer (fc2)** for feature extraction (4096-dimensional feature vectors).

3. **Feature Extraction**

   * Converts each image into a 4096-length feature vector.
   * Captures semantic content like shapes, patterns, and textures.

4. **Dimensionality Reduction (PCA)**

   * Reduces features from 4096 to 300 dimensions.
   * Speeds up computation and removes redundancy.

5. **Similarity Computation**

   * Uses **cosine distance** to measure how close two feature vectors are.
   * Returns top 5 most similar images to the query.

6. **Visualization**

   * Displays the query image along with its most visually similar images.
   * Annotates results with similarity scores.

7. **Adding New Images**

   * Newly added images are processed, embedded in PCA space, and compared to existing features for retrieval.

---

## 🧠 Key Concepts

* **Feature Extraction:** Transforming raw pixel data into semantic feature vectors using CNNs.
* **Transfer Learning:** Leveraging VGG16 trained on ImageNet instead of training from scratch.
* **Dimensionality Reduction:** Using PCA to improve performance and avoid redundancy.
* **Similarity Search:** Using cosine distance to find nearest visual neighbors.

---

## 🧩 Tech Stack

* **Language:** Python
* **Libraries:**

  * `Keras`, `TensorFlow`, `NumPy`, `Matplotlib`
  * `Scikit-learn`, `PIL`, `Scipy`
  * `tqdm`, `pickle`
* **Model:** VGG16 (pretrained on ImageNet)


## 📈 Output Example

The model retrieves top 5 visually similar images with their cosine similarity scores:

| Query Image                | Similar Images                                                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| ![query](assets/query.jpg) | ![similar1](assets/sim1.jpg) ![similar2](assets/sim2.jpg) ![similar3](assets/sim3.jpg) ![similar4](assets/sim4.jpg) ![similar5](assets/sim5.jpg) |

---

## 💾 Saving Extracted Features

Features and PCA components are saved for reuse:

```python
pickle.dump([images, pca_features, pca], open('features_5cats.p', 'wb'))
```

This avoids recomputation when new queries or datasets are added.

---

## 🧩 Applications

* Content-based image retrieval systems (CBIR)
* Visual product recommendation engines
* Duplicate image detection
* Photo organization and tagging tools

---

