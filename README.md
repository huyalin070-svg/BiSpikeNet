# BiSpikeNet
Research on the BiSpikeNet Classification Model for Tea Plant Pests and Diseases
# Overview
- *Our Tea Plant Pests and Diseases*
<img width="722" height="455" alt="image" src="https://github.com/user-attachments/assets/55c26b0b-1931-48b8-b1f6-0f9f6c4933d6" />

- *Our Model*
<img width="1026" height="893" alt="fig5" src="https://github.com/user-attachments/assets/6970dabd-1e02-4af8-94c6-3cae4e9cc969" />

- *Our Algorithm*
<img width="1040" height="732" alt="fig4" src="https://github.com/user-attachments/assets/c88106ac-6df7-433d-80ec-8dc1723aa3e7" />

# environment requirements
torch==2.7.0+cu126
torchvision==0.22.0
numpy==1.26.0
matplotlib==3.8.4
pillow==10.4.0

# installation steps
conda install requirements.txt

# training/inference commands
```python
python Train.py --batch_size 64 --epochs 100 --learning_rate 0.001 --data ./data/TeaLeafBD --save ./Results/exp1
```
# dataset structure description
nl/<br>
├── train/<br>
│   ├── Healthy Tea/<br>
│   ├── Broken tea leaves/<br>
│   ├── Tea leaf grey spot disease/<br>
│   ├── Tea leaf red spot disease/<br>
│   ├── Tea leaf spot disease/<br>
│   ├── Tea leaf brown spot disease/<br>
│   ├── Tea leafhopper disease/<br>
├── val/<br>
│   ├── Healthy Tea/<br>
│   ├── Broken tea leaves/<br>
│   ├── Tea leaf grey spot disease/<br>
│   ├── Tea leaf red spot disease/<br>
│   ├── Tea leaf spot disease/<br>
│   ├── Tea leaf brown spot disease/<br>
│   ├── Tea leafhopper disease/<br>
├── test/<br>
│   ├── Healthy Tea/<br>
│   ├── Broken tea leaves/<br>
│   ├── Tea leaf grey spot disease/<br>
│   ├── Tea leaf red spot disease/<br>
│   ├── Tea leaf spot disease/<br>
│   ├── Tea leaf brown spot disease/<br>
│   ├── Tea leafhopper disease/<br>
    
# weights link
Our well-trained model can be found in [Google Drive](https://drive.google.com/file/d/1jGYpS9NOy03UPHKprkupjfJL7Hb4_Dqj/view?usp=drive_link)

# Citing
