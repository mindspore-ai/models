
## Dataset

- - PPM100: test data. The corresponding training dataset is not open.
- Adobe Matting Dataset (AMD): Please contact Brian Price (bprice@adobe.com) for the dataset.
- [P3M-10k](https://github.com/JizhiziLi/P3M). We used this dataset at the recommendation of the original author, and the dataset can be downloaded from this link. The directory is as follows.

```shell
P3M-10k
├── train
    ├── blurred_image
    ├── mask (alpha mattes)
├── validation
    ├── P3M-500-P
        ├── blurred_image
        ├── mask
        ├── trimap
    ├── P3M-500-NP
        ├── original_image
        ├── mask
        ├── trimap
```

