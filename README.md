<div align="center">
<h1>MuCoNet</h1>
<h3>MuCoNet: Multiscale Context Grouping and Detail Compensation with Global Modulation for Polyp Segmentation</h3>
</div>


## 🛠Setup
#### 1. Environment

```html
conda create -n gmamba-polyp python=3.10
conda activate gmamba-polyp
cd GMambaPolyp
pip install -r requirements.txt
```

## 📚Data Preparation

Downloading training and testing datasets and move them into `./data/`.

**TrainDatasets**: The dataset can be found [here](https://drive.google.com/drive/folders/1NVEDXDeIvKHw55dOnL6CbbbsiWrg41FH?usp=drive_link).

**TestDatasets**: The dataset can be found [here](https://drive.google.com/drive/folders/12i58jDzDGE8MiQ-QxPxiltbX8GkzwaG4?usp=drive_link).

```html
GMambaPolyp
├── data
    ├── TrainDataset
        ├── images
        ├── masks
        ├── edges
    ├── TestDataset
        ├── Kvasir
            ├── images
            ├── masks
            ├── edges
        ├── CVC-ClinicDB
        ├── CVC-300
        ├── CVC-ColonDB
        ├── ETIS-LaribPolypDB
```

## ⏳Training

```html
python train.py
```

## 🔖Testing

```html
python test.py
```