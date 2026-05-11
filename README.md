# BiSpikeNet
Research on the BiSpikeNet Classification Model for Tea Plant Pests and Diseases

# environment requirements
torch==2.7.0+cu126
torchvision==0.22.0
numpy==1.26.0
matplotlib==3.8.4
pillow==10.4.0

# installation steps
conda install requirements.txt

# training/inference commands
python Train.py --batch_size 64 --epochs 100 --learning_rate 0.001 --data ./data/TeaLeafBD --save ./Results/exp1

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
https://drive.google.com/file/d/1jGYpS9NOy03UPHKprkupjfJL7Hb4_Dqj/view?usp=drive_link
