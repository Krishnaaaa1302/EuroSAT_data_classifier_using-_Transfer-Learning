# 🛰️ EuroSAT Land-Use Classification with Transfer Learning  
  
A deep learning project that classifies Sentinel-2 satellite imagery into 10 land-use categories using a **ResNet-50 CNN pretrained on ImageNet**, fine-tuned in **PyTorch**. Achieves **~97–98% test accuracy** through transfer learning, data augmentation, and layer-wise fine-tuning.  
  
---  
  
## 📌 Overview  

	
Dataset	EuroSAT — 27,000 labeled Sentinel-2 images
Classes	10 (Forest, River, Residential, Industrial, Highway, Pasture, etc.)
Image Size	64×64 (upsampled to 224×224 for the backbone)
Model	ResNet-50 (ImageNet-pretrained)
Framework	PyTorch + torchvision
Best Accuracy	~98% (test)
🧠 Approach

This project uses transfer learning in two phases:

    Feature Extraction (Phase 1) — Freeze the ResNet-50 backbone and train only a new classifier head. Reaches ~90–95% test accuracy quickly.

    Fine-Tuning (Phase 2) — Unfreeze the final residual block (layer4) and train it with a lower learning rate to adapt high-level features to satellite imagery. Pushes accuracy to ~97–98%.

Key Techniques

    ImageNet normalization for compatibility with pretrained weights

    Data augmentation (random flips, rotations) to improve generalization

    Dropout regularization in the classifier head

    Layer-wise discriminative learning rates during fine-tuning

    Grad-CAM for model interpretability

📁 Project Structure

eurosat-classifier/  
├── eurosat_classifier.py    # Main training + evaluation script  
├── eurosat_resnet50.pth     # Saved model weights (generated)  
├── confusion_matrix.png     # Per-class performance heatmap (generated)  
├── gradcam.png              # Interpretability visualization (generated)  
├── requirements.txt  
└── README.md  

⚙️ Installation

bash
Copy
git clone https://github.com/<your-username>/eurosat-classifier.git  
cd eurosat-classifier  
pip install -r requirements.txt  

requirements.txt

torch  
torchvision  
scikit-learn  
matplotlib  
seaborn  
opencv-python  

🚀 Usage

bash
Copy
python eurosat_classifier.py  

The script will automatically:

    Download the EuroSAT dataset

    Train the classifier head (Phase 1)

    Fine-tune layer4 (Phase 2)

    Save the model and generate evaluation plots

    💡 A GPU is strongly recommended. Google Colab (free tier) works well.

📊 Results
Stage	Train Acc	Test Acc
Phase 1 (frozen backbone)	~94%	~93–96%
Phase 2 (fine-tuned layer4)	~99%	~97–98%
Outputs

    confusion_matrix.png — Per-class precision/recall visualization

    gradcam.png — Heatmap showing which regions drove the model's prediction

    Console: full classification report (precision, recall, F1 per class)

🔍 Interpretability (Grad-CAM)

Grad-CAM overlays highlight the image regions the model focused on for its prediction — confirming it relies on meaningful features (e.g., vegetation texture for Forest) rather than artifacts.
🔮 Possible Extensions

    Compare backbones (EfficientNet, MobileNet, ViT)

    Full end-to-end fine-tuning for ~99% accuracy

    Export to ONNX for deployment

    Build a Streamlit web demo

    Experiment tracking with Weights & Biases

📚 References

    EuroSAT Dataset (Helber et al., 2019)

    Deep Residual Learning (He et al., 2015)

    Grad-CAM (Selvaraju et al., 2017)
