# AI-Powered COVID-19 Detection Using Transfer Learning and Grad-CAM Visualization

## Overview
This project leverages deep learning to detect COVID-19 from chest X-ray images. It utilizes **transfer learning** with pre-trained CNNs for efficient model training and **Grad-CAM visualization** for interpretability. The model classifies chest X-rays into:
- COVID-19 Positive
- Normal
- Other Lung Infections (Viral Pneumonia and Lung Opacity)

The objective is to develop a reliable AI tool to assist researchers and medical professionals.

---

## Dataset

### COVID-19 Chest X-Ray Database
A dataset collaboratively developed by Qatar University, the University of Dhaka, and other researchers, with quality assured by medical professionals.

#### Distribution:
| **Category**             | **Number of Images** |
|---------------------------|-----------------------|
| COVID-19 Positive         | 3,616                |
| Normal Cases              | 10,192               |
| Lung Opacity              | 6,012                |
| Viral Pneumonia           | 1,345                |

#### Details:
- **COVID-19 Images**:
  - PadChest (2,473 images)
  - Germany medical school (183 images)
  - SIRM, GitHub, Kaggle, Twitter (559 images)
  - Other GitHub sources (400 images)
  
- **Normal Images**:
  - RSNA (8,851 images)
  - Kaggle (1,341 images)
  
- **Lung Opacity Images**:
  - RSNA (6,012 images)

- **Viral Pneumonia Images**:
  - Kaggle's Pneumonia dataset (1,345 images)

#### Format:
- **File Format**: PNG
- **Resolution**: 299x299 pixels

#### Download Programmatically:
```python
import kagglehub

# Download the COVID-19 Radiography Dataset
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")

print("Dataset path:", path)

```
## Methodology

### Data Preprocessing
- **Image Resizing**: All images resized to 224x224 pixels.
- **Normalization**: Pixel values scaled between 0 and 1.
- **Augmentation**:
  - Rotation (up to 40 degrees)
  - Width and height shifts
  - Zooming
  - Horizontal flipping


Transfer Learning
Four pre-trained CNNs were used for feature extraction and classification:
1. **DenseNet-121**: Achieved the best performance with an accuracy of 96.49%.
2. **ResNet-50**
3. **MobileNet**
4. **VGG16**

Key steps:
- Freeze the pre-trained layers except for the last two.
- Add dense layers with batch normalization and dropout for fine-tuning.
- Train the models using 70% of the dataset and validate on 30%.

Grad-CAM Visualization
Grad-CAM (Gradient-weighted Class Activation Mapping) was used to interpret the predictions. It highlights the critical regions in chest X-rays that influence the model's classification.

Results
Performance Metrics:
| Model        | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score |
|--------------|--------------|---------------|------------|-----------------|----------|
| DenseNet-121 | 96.49        | 93.45         | 100        | 92.99           | 0.97     |
| MobileNet    | 96.48        | 86.93         | 100        | 92.99           | 0.97     |
| ResNet-50    | 92.48        | 86.93         | 100        | 84.97           | 0.93     |
| VGG16        | 83.27        | 76.80         | 95.51      | 70.96           | 0.85     |

DenseNet-121 and MobileNet demonstrated the highest accuracy, with Grad-CAM visualizations offering improved interpretability.

Implementation
Clone the Repository
```python
git clone https://github.com/kunal28-07/AI-Powered-COVID-19-Detection-Using-Transfer-Learning-and-Grad-CAM-Visualization.git
cd AI-Powered-COVID-19-Detection-Using-Transfer-Learning-and-Grad-CAM-Visualization
```

Install Dependencies
```python
pip install -r requirements.txt
```
Train the Model
Run the Jupyter Notebook kunal-raj-deep-learning.ipynb to train the model on the dataset. Adjust parameters such as batch size, learning rate, and number of epochs as needed.

Visualize Grad-CAM
The notebook includes code to visualize Grad-CAM heatmaps for both correct and misclassified predictions.

Citation
If you use this project or dataset in your research, please cite the following articles:
1. M.E.H. Chowdhury, T. Rahman, A. Khandakar, et al., “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
2. Rahman, T., Khandakar, A., Qiblawey, Y., et al., “Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images.” arXiv preprint arXiv:2012.02238.

References
- [PadChest Dataset](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711)
- [RSNA Pneumonia Dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
- [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

Contact
For questions, suggestions, or contributions, please open an issue or create a pull request in the GitHub repository.



