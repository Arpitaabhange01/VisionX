# VisionX - CIFAR-10 Image Classification

VisionX is a sleek and interactive web application that classifies images into one of the 10 CIFAR-10 categories using a Convolutional Neural Network (CNN). Built with TensorFlow and Streamlit, it brings deep learning to life with real-time predictions, and performance insights.

---

## 🚀 Features

- 🔍 Upload any image and get instant predictions  
- 🧠 CNN model trained on CIFAR-10 dataset  
- 📊 Confidence score and class-wise performance metrics  

---

## 🧠 What is CIFAR-10?

CIFAR-10 is a dataset of 60,000 32×32 color images across 10 classes:
- ✈️ Airplane  
- 🚗 Automobile  
- 🐦 Bird  
- 🐱 Cat  
- 🦌 Deer  
- 🐶 Dog  
- 🐸 Frog  
- 🐴 Horse  
- 🚢 Ship  
- 🚚 Truck  

Each class has 6,000 images, making it ideal for training image classification models.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, HTML/CSS  
- **Backend**: TensorFlow, Keras  
- **Data**: CIFAR-10 via `tensorflow.keras.datasets`  

---

## 📸 Screenshots

Here are some previews of the project in action:

### 🔹 Home Page
![Home Page](screenshots/vision_01.png)

### 🔹 Image Upload , Prediction & confidence score
![Prediction](screenshots/vision_02.png)


---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Arpitaabhange01/VisionX.git
cd VisionX

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
