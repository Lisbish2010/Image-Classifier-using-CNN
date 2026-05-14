Image Classifier using CNN 🎮🖼️

A deep learning based Image Classifier GUI Application built with TensorFlow, OpenCV, and Tkinter.
This project trains a Convolutional Neural Network (CNN) using the CIFAR-10 dataset and allows users to classify custom images through a modern desktop GUI.

Inspired by clean and fun project presentations similar to classic game repositories like Mario projects.

🚀 Features
Train a CNN model on CIFAR-10 dataset
User-friendly GUI built with Tkinter
Upload and classify custom images
Displays:
Predicted class
Confidence score
Top 3 predictions
Real-time training progress bar
Model save/load functionality
Data augmentation support
Automatic SSL dataset download fix
🧠 Classes Supported

The model can classify images into:

✈️ Airplane
🚗 Automobile
🐦 Bird
🐱 Cat
🦌 Deer
🐶 Dog
🐸 Frog
🐴 Horse
🚢 Ship
🚚 Truck
🛠️ Technologies Used
Technology	Purpose
Python	Core programming
TensorFlow / Keras	Deep Learning
OpenCV	Image processing
Tkinter	GUI interface
NumPy	Numerical operations
Pillow (PIL)	Image handling
Matplotlib	Visualization
📂 Project Structure
Image-Classifier/
│
├── main.py
├── cifar10_model.h5
├── README.md
└── requirements.txt
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/Image-Classifier.git
cd Image-Classifier
2️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Project
python main.py
🏋️ Train the Model

If no trained model exists:

Open the application
Click "Train New Model"
Wait for training to complete
Model will automatically save as:
cifar10_model.h5
📸 How to Use
Launch the application
Click Browse Image
Select any image
Click Classify Image
View predictions instantly
🧩 CNN Architecture

The project uses:

Multiple Conv2D layers
Batch Normalization
MaxPooling
Dropout regularization
Dense neural layers
Softmax output layer
📈 Expected Accuracy

Typical validation accuracy:

~75% to 85%

Depending on training environment and epochs.

📦 Requirements

Create a requirements.txt file with:

tensorflow
opencv-python
numpy
matplotlib
pillow
💡 Future Improvements
Add webcam live prediction
Support custom datasets
Add drag & drop upload
Improve UI animations
Export prediction reports
🖼️ Screenshot

Add your project screenshot here
