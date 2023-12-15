# Scene-STGCNN
code for "Scene-constrained spatial-temporal graph convolutional network for pedestrian trajectory prediction"

# Training

```bash
python train.py --dataset eth --tag scene-stgcnn-eth

python train.py --dataset hotel --tag scene-stgcnn-hotel

python train.py --dataset univ --tag scene-stgcnn-univ

python train.py --dataset zara1 --tag scene-stgcnn-zara1

python train.py --dataset zara2 --tag scene-stgcnn-zara2
```

# Test

```bash
python test.py
```