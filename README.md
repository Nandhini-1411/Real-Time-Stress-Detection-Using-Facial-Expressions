# Facial Emotion Recognition with VGG16 & FER2013

This project focuses on **Facial Emotion Recognition** using the **FER2013 dataset** and **VGG16** as the base model. The trained model can classify emotions into seven categories:  
**Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.**

The model is trained incrementally with **early stopping, learning rate reduction, and model checkpointing** to optimize performance.


## 🚀 Features
- Uses **VGG16** as the backbone model with **transfer learning**.
- Trains in **increments of 20 epochs** for better convergence.
- Implements **data augmentation** for improved generalization.
- Uses **FER2013 dataset** for training.
- Saves and loads the **best model automatically**.


## 🔥 Dataset: FER2013
The **FER2013 dataset** contains **35,887 grayscale images (48x48)** categorized into **7 emotions**:
- 😡 **Angry**  
- 🤢 **Disgust**  
- 😨 **Fear**  
- 😃 **Happy**  
- 😢 **Sad**  
- 😲 **Surprise**  
- 😐 **Neutral**  

📥 **Download Dataset**: [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  

## 📝 Model Training Process
- Loads VGG16 pre-trained on ImageNet.
- Freezes the first 10 layers to retain pre-learned features.
- Adds Global Average Pooling, Dense, and Dropout layers.
- Uses Adam optimizer with categorical_crossentropy loss.
- Applies data augmentation for robustness.
- Implements early stopping, learning rate reduction, and checkpointing.

💡 Results
- Achieved high accuracy using transfer learning.
-  Fine-tuned VGG16 to adapt to the FER2013 dataset.
- Optimized model with incremental training strategy.
