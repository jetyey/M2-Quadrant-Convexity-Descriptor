# M2-Quadrant-Convexity-Descriptor

## Training
We have Datas/training.csv. The results of training using Weka

## Test
We test its classification capacity in different levels of noises. CSV files are found inside Gauss,S&P and Speckle

## Models
Currently following models are available:
- [`Q Convexity`](M2-Quadrant-Convexity-Descriptor-master/Qconvexity.py)
    - [1] Brunetti, Sara "A Spatial Convexity Descriptor for Object Enlacement" DGCI 2019

## How to use.
### Setup
First, clone the repository and run setup.

```
git clone https://github.com/jetyey/M2-Quadrant-Convexity-Descriptor
cd M2-Quadrant-Convexity-Descriptor-master
```

### Run Program
You can use sample script by 
```
python Qconvexity.py -ri ReferenceImage -i otherObject
```
