Instructions to reproduce reported results.

# CIFAR 10
## 4000 labels
Threshold 0.95 - best_model_cifar10_4000_95.pt - error rate 16.66%
```
& python "./main.py" --dataset cifar10 --num-labeled 4000 --num-validation 1000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.95 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
Threshold 0.75 - best_model_cifar10_4000_75.pt - error rate 17.96%
```
& python "./main.py" --dataset cifar10 --num-labeled 4000 --num-validation 1000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.75 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
Threshold 0.6 - best_model_cifar10_4000_60.pt - error rate 17.25%
```
& python "./main.py" --dataset cifar10 --num-labeled 4000 --num-validation 1000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.6 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
## 250 labels
Threshold 0.95 - best_model_cifar10_250_95.pt - error rate 63.72%
```
& python "./main.py" --dataset cifar10 --num-labeled 250 --num-validation 100 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.95 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_250/"
```
Threshold 0.75 - best_model_cifar10_250_75.pt - error rate 65.01%
```
& python "./main.py" --dataset cifar10 --num-labeled 250 --num-validation 100 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.75 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_250/"
```
Threshold 0.6 - best_model_cifar10_250_60.pt - error rate 61.96%
```
& python "./main.py" --dataset cifar10 --num-labeled 250 --num-validation 100 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.6 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_250/"
```
# CIFAR 100
## 10000 labels
Threshold 0.95 - best_model_cifar100_10000_95.pt - error rate 46.20%
```
& python "./main.py" --dataset cifar100 --num-labeled 10000 --num-validation 2000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.95 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
Threshold 0.75 - best_model_cifar100_10000_75.pt - error rate 43.54%
```
& python "./main.py" --dataset cifar100 --num-labeled 10000 --num-validation 2000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.75 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
Threshold 0.6 - best_model_cifar100_10000_60.pt - error rate 47.30%
```
& python "./main.py" --dataset cifar100 --num-labeled 10000 --num-validation 2000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.6 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
## 4000 labels
Threshold 0.95 - best_model_cifar100_4000_95.pt - error rate 64.12%
```
& python "./main.py" --dataset cifar100 --num-labeled 4000 --num-validation 1000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.95 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
Threshold 0.75 - best_model_cifar100_4000_75.pt - error rate 64.91%
```
& python "./main.py" --dataset cifar100 --num-labeled 4000 --num-validation 1000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.75 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```
Threshold 0.6 - best_model_cifar100_4000_60.pt - error rate 64.89%
```
& python "./main.py" --dataset cifar100 --num-labeled 4000 --num-validation 1000 --lr 0.01 --momentum 0.95 --wd 0.0005 --threshold 0.6 --alpha 3 --t1 5120 --t2 76800 --modelpath "./models/obs/cifar10_4000/"
```