# LFDBP-Net

> **Note**: The paper is currently under submission. Once the paper is accepted, the training code will be made available as open-source. Stay tuned for updates! 🚀
## 📥 Download Datasets and Pretrained Models

Download our CSRD-CR dataset and pretrained model weights from Baidu Cloud Disk:

🔗 **Link**: [https://pan.baidu.com/s/1sELBm-ZTP7d0MK5oudS_1g?pwd=k5vg](https://pan.baidu.com/s/1sELBm-ZTP7d0MK5oudS_1g?pwd=k5vg)  
🔐 **Password**: `k5vg`


---

## ☁️️ Run Test

Use the following commands to test different datasets:

### 🔹 CSRD-CR

```bash
python test_CSRD.py 
  --dataset_root ./dataset/CSRD \
  --level [easy | medium | hard] \
  --season [spring | summer | fall | winter]
```

**Parameters**:
- `--dataset_root`: Root directory of CSRD dataset  
- `--level`: Difficulty level (`easy`, `medium`, or `hard`)  
- `--season`: Season of the dataset (`spring`, `summer`, `fall`, or `winter`)


### 🔹 RICE1

```bash
python test_RICE1.py --dataset_root ./dataset/CSRD
```

### 🔹 RICE2

```bash
python test_RICE2.py --dataset_root ./dataset/CSRD
```

### 🔹 TCD

```bash
python test_TCD.py --dataset_root ./dataset/CSRD
```

## 📸 Qualitative Results

### 🔹 **Results on CSRD-Easy Challenge Testing Images**
![CSRD-Easy Image 1](visual_image/CSRD-Easy.jpg)


### 🔹 **Results on CSRD-Medium Challenge Testing Images**
![CSRD-Medium Image 1](visual_image/CSRD-Medium.jpg)


### 🔹 **Results on CSRD-Hard Challenge Testing Images**
![CSRD-Hard Image 1](visual_image/CSRD-Hard.jpg)


### 🔹 **Results on RICE Challenge Testing Images**
![RICE Image 1](visual_image/RICE.jpg)


### 🔹 **Results on TCD Challenge Testing Images**
![TCD Image 1](visual_image/TCD.jpg)
