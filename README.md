Here's a comprehensive `README.md` file for your coffee leaf disease classification project:

```markdown
# Coffee Leaf Disease Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a deep learning model to classify diseases in coffee leaves using convolutional neural networks (CNN). The model is trained to recognize various coffee leaf conditions from images.

## Features

- Dataset preprocessing and augmentation
- CNN model implementation
- Training with early stopping and model checkpointing
- Model evaluation with classification metrics
- Model export to multiple formats (H5, SavedModel, TFLite, TFJS)
- Inference pipeline for new images

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
└── penyakit kopi/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── validation/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── ...
```

## Usage

1. **Setup**: Mount Google Drive and extract the dataset
2. **Data Preparation**: The script automatically splits the dataset into train/validation/test sets
3. **Model Training**: Run the training pipeline with data augmentation
4. **Evaluation**: View accuracy/loss plots and classification metrics
5. **Inference**: Use the trained model to predict new images

### Training the Model

```python
# The model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Training with callbacks
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=callbacks
)
```

### Making Predictions

```python
img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
predicted_class = class_labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

## Model Export

The trained model is exported in multiple formats:
- H5 (Keras format)
- SavedModel (TensorFlow format)
- TFLite (for mobile deployment)
- TFJS (for web deployment)

## Results

The model provides:
- Training/validation accuracy and loss curves
- Classification report (precision, recall, f1-score)
- Confusion matrix visualization
- Test set evaluation metrics

## Directory Structure

```
project/
├── dataset/                    # Dataset directory
├── submission/                 # Exported models
│   ├── saved_model/            # TensorFlow SavedModel
│   ├── tflite/                 # TFLite model + labels
│   └── tfjs_model/             # TensorFlow.js model
├── requirements.txt            # Dependencies
└── coffee_leaf_classification.ipynb  # Main notebook
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras documentation
- Google Colab for providing computational resources
```

This README provides a comprehensive overview of your project with:
1. Clear badges showing technologies used
2. Installation instructions
3. Dataset structure requirements
4. Usage examples
5. Model architecture details
6. Export information
7. Expected results
8. Project structure
9. License information

You can customize it further by adding:
- Specific accuracy metrics from your model
- Sample images from your dataset
- Deployment instructions
- References to related papers or projects