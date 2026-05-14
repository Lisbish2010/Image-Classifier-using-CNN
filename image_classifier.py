import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import ssl
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

class ImageClassifier:
    def __init__(self):
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.model = None
        self.img_height = 32
        self.img_width = 32
        
    def fix_ssl_certificate(self):
        """Fix SSL certificate verification issues"""
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception as e:
            print(f"Warning: Could not modify SSL settings: {e}")
    
    def load_and_prepare_data(self):
        """Load CIFAR-10 dataset with SSL fix"""
        self.fix_ssl_certificate()
        
        try:
            from tensorflow.keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        except Exception as e:
            raise Exception(f"Error downloading dataset: {e}\n\n"
                          "Please download manually from:\n"
                          "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """Build CNN model for image classification"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=20, progress_callback=None):
        """Train the model with progress callback"""
        self.model = self.build_model()
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        x_train_aug = data_augmentation(x_train, training=True)
        
        # Custom callback for GUI progress
        class GUICallback(keras.callbacks.Callback):
            def __init__(self, progress_callback):
                super().__init__()
                self.progress_callback = progress_callback
                
            def on_epoch_end(self, epoch, logs=None):
                if self.progress_callback:
                    self.progress_callback(epoch + 1, epochs, logs)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=0
            )
        ]
        
        if progress_callback:
            callbacks.append(GUICallback(progress_callback))
        
        history = self.model.fit(
            x_train_aug, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            batch_size=64,
            verbose=0,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, filepath='cifar10_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
    
    def load_model(self, filepath='cifar10_model.h5'):
        """Load a pre-trained model"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            return True
        return False
    
    def preprocess_image(self, image_path):
        """Preprocess user input image using OpenCV"""
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height))
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_rgb
    
    def classify_image(self, image_path):
        """Classify a user-provided image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        img_batch, original_img = self.preprocess_image(image_path)
        predictions = self.model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        class_name = self.class_names[predicted_class]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [(self.class_names[idx], predictions[0][idx] * 100) for idx in top_3_idx]
        
        return class_name, confidence, top_3, original_img


class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier - CIFAR-10")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.classifier = ImageClassifier()
        self.model_path = 'cifar10_model.h5'
        self.current_image_path = None
        
        self.setup_ui()
        self.check_model()
        
    def setup_ui(self):
        """Setup the GUI interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🖼️ Image Classifier", 
                              font=('Arial', 24, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Image display
        left_panel = tk.Frame(content_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        image_label = tk.Label(left_panel, text="Image Preview", 
                              font=('Arial', 12, 'bold'), bg='white')
        image_label.pack(pady=10)
        
        self.image_canvas = tk.Canvas(left_panel, width=400, height=400, bg='#ecf0f1')
        self.image_canvas.pack(pady=10, padx=10)
        
        # Placeholder text
        self.image_canvas.create_text(200, 200, 
                                      text="No Image Loaded\n\nClick 'Browse Image' to start",
                                      font=('Arial', 12), fill='#7f8c8d', tags='placeholder')
        
        # Right panel - Controls and results
        right_panel = tk.Frame(content_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Model status
        status_frame = tk.Frame(right_panel, bg='white')
        status_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(status_frame, text="Model Status:", font=('Arial', 11, 'bold'), 
                bg='white').pack(anchor=tk.W)
        
        self.status_label = tk.Label(status_frame, text="Checking...", 
                                     font=('Arial', 10), bg='white', fg='#e67e22')
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Buttons
        button_frame = tk.Frame(right_panel, bg='white')
        button_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.browse_btn = ttk.Button(button_frame, text="📁 Browse Image", 
                                     command=self.browse_image, state=tk.DISABLED)
        self.browse_btn.pack(fill=tk.X, pady=5)
        
        self.classify_btn = ttk.Button(button_frame, text="🎯 Classify Image", 
                                       command=self.classify_image, state=tk.DISABLED)
        self.classify_btn.pack(fill=tk.X, pady=5)
        
        self.train_btn = ttk.Button(button_frame, text="🚀 Train New Model", 
                                    command=self.train_model)
        self.train_btn.pack(fill=tk.X, pady=5)
        
        # Results area
        results_frame = tk.Frame(right_panel, bg='white')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        tk.Label(results_frame, text="Classification Results:", 
                font=('Arial', 11, 'bold'), bg='white').pack(anchor=tk.W)
        
        self.results_text = tk.Text(results_frame, height=15, font=('Courier', 10),
                                   bg='#ecf0f1', relief=tk.FLAT, padx=10, pady=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.results_text.insert('1.0', "No classification results yet.\n\n"
                                       "Supported classes:\n" + 
                                       ", ".join(self.classifier.class_names))
        self.results_text.config(state=tk.DISABLED)
        
        # Progress bar (hidden initially)
        self.progress_frame = tk.Frame(right_panel, bg='white')
        self.progress_label = tk.Label(self.progress_frame, text="Training Progress:", 
                                       font=('Arial', 10), bg='white')
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', 
                                           length=300)
        self.progress_status = tk.Label(self.progress_frame, text="", 
                                       font=('Arial', 9), bg='white')
        
    def check_model(self):
        """Check if model exists and load it"""
        if os.path.exists(self.model_path):
            try:
                self.classifier.load_model(self.model_path)
                self.status_label.config(text="✓ Model loaded and ready", fg='#27ae60')
                self.browse_btn.config(state=tk.NORMAL)
                self.update_results("Model loaded successfully!\n\n"
                                  "You can now classify images.\n\n"
                                  "Supported classes:\n" + 
                                  ", ".join(self.classifier.class_names))
            except Exception as e:
                self.status_label.config(text=f"✗ Error loading model: {str(e)}", fg='#e74c3c')
        else:
            self.status_label.config(text="⚠ No trained model found. Please train one.", 
                                   fg='#e67e22')
            self.update_results("No pre-trained model found.\n\n"
                              "Click 'Train New Model' to create one.\n\n"
                              "This will take 10-20 minutes.")
    
    def browse_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.config(state=tk.NORMAL)
    
    def display_image(self, image_path):
        """Display the selected image on canvas"""
        try:
            # Load and resize image for display
            img = Image.open(image_path)
            
            # Calculate resize dimensions maintaining aspect ratio
            max_size = 380
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Clear canvas
            self.image_canvas.delete('all')
            
            # Display image centered
            x = (400 - img.width) // 2
            y = (400 - img.height) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.image_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def classify_image(self):
        """Classify the selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            # Disable buttons during classification
            self.classify_btn.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.DISABLED)
            self.update_results("Classifying image...\n\nPlease wait...")
            self.root.update()
            
            # Classify
            class_name, confidence, top_3, _ = self.classifier.classify_image(
                self.current_image_path
            )
            
            # Display results
            result_text = f"{'='*40}\n"
            result_text += f"CLASSIFICATION RESULT\n"
            result_text += f"{'='*40}\n\n"
            result_text += f"🎯 Predicted Class: {class_name.upper()}\n"
            result_text += f"📈 Confidence: {confidence:.2f}%\n\n"
            result_text += f"{'='*40}\n"
            result_text += f"🏆 Top 3 Predictions:\n"
            result_text += f"{'='*40}\n\n"
            
            for i, (cls, conf) in enumerate(top_3, 1):
                result_text += f"{i}. {cls:12s} - {conf:6.2f}%\n"
            
            self.update_results(result_text)
            
            # Re-enable buttons
            self.classify_btn.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.browse_btn.config(state=tk.NORMAL)
    
    def update_results(self, text):
        """Update the results text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', text)
        self.results_text.config(state=tk.DISABLED)
    
    def train_model(self):
        """Train a new model in a separate thread"""
        response = messagebox.askyesno(
            "Train Model",
            "Training will take 10-20 minutes and use significant CPU/GPU resources.\n\n"
            "Do you want to continue?"
        )
        
        if not response:
            return
        
        # Disable all buttons during training
        self.browse_btn.config(state=tk.DISABLED)
        self.classify_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        
        # Show progress bar
        self.progress_frame.pack(fill=tk.X, padx=15, pady=10)
        self.progress_label.pack(anchor=tk.W)
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_status.pack(anchor=tk.W)
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self.train_model_thread)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model_thread(self):
        """Training thread function"""
        try:
            self.update_results("Loading CIFAR-10 dataset...\n\n"
                              "This may take a few minutes on first run.")
            
            # Load data
            (x_train, y_train), (x_test, y_test) = self.classifier.load_and_prepare_data()
            
            self.update_results(f"Dataset loaded!\n\n"
                              f"Training samples: {x_train.shape[0]}\n"
                              f"Test samples: {x_test.shape[0]}\n\n"
                              f"Starting training...")
            
            # Train model
            history = self.classifier.train_model(
                x_train, y_train, x_test, y_test, 
                epochs=15,
                progress_callback=self.training_progress
            )
            
            # Save model
            self.classifier.save_model(self.model_path)
            
            # Evaluate
            test_loss, test_acc = self.classifier.model.evaluate(
                x_test, y_test, verbose=0
            )
            
            # Update UI on main thread
            self.root.after(0, self.training_complete, test_acc)
            
        except Exception as e:
            self.root.after(0, self.training_error, str(e))
    
    def training_progress(self, epoch, total_epochs, logs):
        """Update progress during training"""
        progress = (epoch / total_epochs) * 100
        
        def update_ui():
            self.progress_bar['value'] = progress
            self.progress_status.config(
                text=f"Epoch {epoch}/{total_epochs} - "
                     f"Accuracy: {logs.get('accuracy', 0)*100:.2f}% - "
                     f"Val Accuracy: {logs.get('val_accuracy', 0)*100:.2f}%"
            )
            self.update_results(
                f"Training in progress...\n\n"
                f"Epoch: {epoch}/{total_epochs}\n"
                f"Training Accuracy: {logs.get('accuracy', 0)*100:.2f}%\n"
                f"Validation Accuracy: {logs.get('val_accuracy', 0)*100:.2f}%\n"
                f"Loss: {logs.get('loss', 0):.4f}\n\n"
                f"Please wait..."
            )
        
        self.root.after(0, update_ui)
    
    def training_complete(self, test_acc):
        """Called when training is complete"""
        # Hide progress bar
        self.progress_frame.pack_forget()
        
        # Update status
        self.status_label.config(text="✓ Model trained and ready", fg='#27ae60')
        
        # Re-enable buttons
        self.browse_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        
        # Show results
        self.update_results(
            f"{'='*40}\n"
            f"TRAINING COMPLETE!\n"
            f"{'='*40}\n\n"
            f"✓ Model saved successfully\n"
            f"✓ Test Accuracy: {test_acc*100:.2f}%\n\n"
            f"You can now classify images!"
        )
        
        messagebox.showinfo(
            "Training Complete",
            f"Model trained successfully!\n\nTest Accuracy: {test_acc*100:.2f}%"
        )
    
    def training_error(self, error_msg):
        """Called when training fails"""
        # Hide progress bar
        self.progress_frame.pack_forget()
        
        # Re-enable train button
        self.train_btn.config(state=tk.NORMAL)
        
        # Show error
        self.update_results(f"Training failed:\n\n{error_msg}")
        messagebox.showerror("Training Error", f"Training failed:\n\n{error_msg}")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()