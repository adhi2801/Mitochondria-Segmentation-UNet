# Mitochondria-Segmentation-UNet  

This project implements a UNet-based deep learning model for segmenting mitochondria in electron microscopy images. The model is trained using patchified images and data augmentation to improve segmentation accuracy.  

## Project Overview  

- Implements **UNet architecture** for precise mitochondria segmentation.  
- Uses **patchification** to divide large images into **256x256** patches.  
- Applies **data augmentation** techniques to enhance training data.  
- Trains the model on **electron microscopy images** with corresponding masks.  
- Evaluates performance using **IoU (Intersection over Union) score**.  

## Project Structure  

Mitochondria-Segmentation-UNet
├── DL_Project_with_UNET.ipynb # Jupyter Notebook with full implementation
├── dataset/ # Folder containing images and masks
├── model/ # Trained UNet model
├── Weights/ # Folder storing trained model weights
├── README.md # Project documentation
├── requirements.txt # List of dependencies


## Setup Instructions  
### Clone the Repository  

git clone https://github.com/your-username/Mitochondria-Segmentation-UNet.git cd Mitochondria-Segmentation-UNet
### Install Dependencies  

pip install -r requirements.txt
### Run the Jupyter Notebook  

jupyter notebook
- Open `DL_Project_with_UNET.ipynb` and run the cells step by step.

## Dataset  

- The dataset consists of **165 electron microscopy images (768x1024)**.  
- Each image is **patchified into 12 smaller images (256x256)**.  
- Corresponding segmentation masks are also patchified for training.  
- Data augmentation includes **rotation, shifting, zooming, flipping, shear transformations**.  

## Model Details  

- **UNet architecture** with encoder-decoder structure.  
- Uses **convolutional layers, batch normalization, and dropout** for feature extraction.  
- **Binary cross-entropy loss** and **Adam optimizer** are used for training.  
- **Early stopping and learning rate reduction** applied to optimize training.  

## Performance Metrics  

| Epoch | Training Accuracy | Validation Accuracy | IoU Score |  
|--------|------------------|--------------------|----------|  
| 1      | 93.50%           | 90.85%             | -        |  
| 5      | 97.83%           | 92.00%             | -        |  
| 10     | 98.25%           | 98.45%             | -        |  
| 20     | 98.71%           | 98.67%             | -        |  
| 25     | 98.73%           | 98.82%             | **0.8897** |  

## Model Training  

- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam (learning rate = 1e-4)  
- **Epochs:** 25  
- **Batch Size:** 4  
- **Validation Split:** 25%  

## Prediction and Evaluation  

- **IoU Score Calculation:** The model achieved an **IoU of 0.8897** on test data.  
- The trained model is saved and can be **loaded for further inference**.  
- Sample visualization of predictions includes:  
  - Input test image  
  - Ground truth mask  
  - Model-generated segmentation output  

## References  

- [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)  
- [Deep Learning for Semantic Segmentation](https://arxiv.org/abs/2001.05566)  
- [UNet Implementation in Keras](https://github.com/zhixuhao/unet)  

---

Replace **`your-username`** with your actual GitHub username before pushing the repository.  

Let me know if you need any modifications.  
