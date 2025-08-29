# Adversarial Attacks and Defense Strategies on Image Classification
A comprehensive study and implementation of adversarial attacks and defense mechanisms on deep neural networks for image classification tasks using ResNet-50 and MobileNetV2 architectures.

📋 Table of Contents

About
Features
Tech Stack
Installation
Usage
Attack Methods
Defense Strategies
Results
Project Structure
Contributing
Authors
References

🎯 About
This project explores the vulnerability of deep neural networks (DNNs) to adversarial attacks in image classification tasks. We investigate how small, imperceptible perturbations can deceive state-of-the-art models like ResNet-50 and MobileNetV2, causing them to misclassify images with high confidence.
The research encompasses:

Adversarial Attack Generation: Implementation of FGSM, DeepFool, and Black Box attacks
Defense Mechanisms: Evaluation of various defense strategies including adversarial training, input transformations, and architectural modifications
Transferability Analysis: Study of how adversarial examples transfer across different neural network architectures
Robustness Evaluation: Comprehensive assessment of model performance under adversarial conditions

✨ Features

Multiple Attack Methods: Implementation of FGSM, Black Box attacks, and other state-of-the-art adversarial techniques
Dual Architecture Support: Testing on both ResNet-50 and MobileNetV2 models
Defense Strategy Evaluation: Assessment of brute-force training, input transformations, and architectural modifications
Comprehensive Analysis: Detailed evaluation of attack transferability and model robustness
Visualization Tools: Clear visualization of adversarial perturbations and their effects
Benchmarking: Performance comparison across different epsilon values and attack strengths

🛠️ Tech Stack
Deep Learning Frameworks:

TensorFlow 2.x - Core framework for model implementation
Keras - High-level neural network API

Data Processing:

NumPy - Numerical computing and array operations
PIL (Python Imaging Library) - Image processing and manipulation
Matplotlib - Data visualization and result plotting

Dataset:

ImageNet - Large-scale image classification dataset with 1000+ classes

Programming Language:

Python 3.7+ - Primary development language

🚀 Installation
Prerequisites

Python 3.7 or higher
TensorFlow 2.x
CUDA-compatible GPU (recommended for faster training)

Setup Instructions

Clone the repository
bashgit clone https://github.com/BhumikaChopra3/adversarial-attacks-defense.git
cd adversarial-attacks-defense

Create virtual environment
bashpython -m venv adversarial_env
source adversarial_env/bin/activate  # On Windows: adversarial_env\Scripts\activate

Install dependencies
bashpip install -r requirements.txt

Download pre-trained models
bash# Models will be automatically downloaded when first running the scripts
# ResNet-50 and MobileNetV2 weights from ImageNet

Prepare dataset
bash# Place your test images in the data/ directory
# Or use the sample images provided


💡 Usage
Basic Usage

Generate Adversarial Examples
bashpython generate_adversarial.py --model resnet50 --attack fgsm --epsilon 0.1

Evaluate Model Robustness
bashpython evaluate_robustness.py --model mobilenetv2 --defense adversarial_training

Compare Attack Methods
bashpython compare_attacks.py --image path/to/image.jpg


Advanced Usage
Custom Attack Generation:
pythonfrom adversarial_attacks import FGSMAttack
from models import load_model

# Load pre-trained model
model = load_model('resnet50')

# Initialize attack
attack = FGSMAttack(model, epsilon=0.1)

# Generate adversarial example
adv_image = attack.generate(original_image, target_label)
Defense Strategy Testing:
pythonfrom defense_strategies import AdversarialTraining
from models import ResNet50Defense

# Initialize defense
defense = AdversarialTraining(epochs=10, attack_strength=0.1)

# Train robust model
robust_model = defense.train(base_model, training_data)
🔓 Attack Methods
White Box Attacks

FGSM (Fast Gradient Sign Method): Single-step attack using gradient information
LFGS (Local Feature Gradients Sign): Targets specific image features

Black Box Attacks

Substitution Model: Uses surrogate models to generate transferable attacks
Zeroth Order Optimization: Gradient-free optimization approach

🛡️ Defense Strategies
1. Brute Force Training (Adversarial Training)

Incorporates adversarial examples during training
Provides regularization effect
Effective against known attack methods

2. Input Transformations

JPEG Compression: Reduces perturbation effects
Principal Component Analysis (PCA): Dimensionality reduction defense

3. Architectural Modifications

Trapdoored Models: Uses honeypots to detect adversarial examples
Enhanced ResNet-50: Modified architecture for improved robustness
MobileNetV2 Defense: Leverages depthwise separable convolutions

📊 Results
Model Performance Comparison
ModelClean AccuracyFGSM (ε=0.1)FGSM (ε=0.3)Black BoxResNet-5094.0%11.5%1.2%37.2%MobileNetV292.3%14.9%2.8%41.5%ResNet-50 + Defense91.2%67.8%45.3%78.9%MobileNetV2 + Defense89.7%72.1%52.6%81.2%
Key Findings

Small perturbations (ε=0.1) significantly reduce model accuracy
Defense mechanisms improve robustness but may slightly reduce clean accuracy
Transferability varies across different model architectures
MobileNetV2 shows better inherent robustness than ResNet-50

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

👥 Authors

Bhumika Chopra (35414813120) - @BhumikaChopra3
Siffatjot Singh (04614813120)

Supervisor: Ms. Sapna Gupta, Assistant Professor, MAIT
📄 License
This project is licensed under the MAIT License - see the LICENSE file for details.
🙏 Acknowledgments

📚 References
Key references include works by Szegedy et al. on adversarial examples, Goodfellow et al. on FGSM, and recent advances in defense strategies. Full bibliography available in the project report.
🚀 Future Work

Implementation of more sophisticated attack methods (PGD, C&W)
Exploration of certified robustness techniques
Investigation of quantum computing threats to adversarial robustness
Development of universal defense mechanisms
Real-world deployment considerations for critical applications


⭐️ If you found this project helpful, please consider giving it a star! For questions or collaboration opportunities, feel free to reach out.
