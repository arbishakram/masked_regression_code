# Masked Regression
MR - Python Implementation

This repositery provides a python implementation of MR (Masked Regression). MR can efficiently synthesize facial expressions.
The demo video for MR can be found here.

### [Masked Linear Regression for Learning Local Receptive Fields for Facial Expression Synthesis](https://link.springer.com/article/10.1007%2Fs11263-019-01256-3)
#### Nazar Khan<sup>1</sup> · Arbish Akram<sup>1</sup> · Arif Mahmood<sup>2</sup> · Sania Ashraf<sup>1</sup>· Kashif Murtaza<sup>1</sup>


<sup>1</sup> Punjab University College of Information Technology
(PUCIT), Lahore, Pakistan <br>
<sup>2</sup> Department of Computer Science, Information Technology
University (ITU), Lahore, Pakistan <br> 
International Journal of Computer Vision (IJCV), Nov 2019.

# Usage

##### 1. Download any Facial Expression Synthesis dataset 
##### 2. Create a folder structure as described [here.](https://github.com/arbishakram/masked_regression_code/blob/master/images/folder_structure.png)

 - Split images into training and test sets (e.g., 90\%/10\% for training and test, respectively).  
 - Crop all images to 128 x 128, where the faces are centered.

##### 3. Training
To train MR:

```bash
$ python main.py --mode train --train_dataset_dir 'dataset/train/'  --image_size 128 --total_images 200 --input_ch 1 
                        --receptive_field 3 --lamda 0.4 
```
##### 4. Test 
To test MR:
```bash
$ python main.py --mode test --test_dataset_dir 'dataset/test/' --image_size 128 --total_images 20 --input_ch 1 
                        --receptive_field 3 
```

##### 5. Test in the wild
To test MR:
```bash
$ python main.py --mode test_inthewild --test_dataset_dir 'dataset/inthewild/' --image_size 128 --total_images 20 --input_ch 1 
                        --receptive_field 3 
```


# Results
Facial expression synthesis on sketches and animals
![Figure 1](https://github.com/arbishakram/masked_regression_code/blob/master/images/Fig_1.png)

Facial expression synthesis on in the wild images
<p align="center"><img width="100%" src="https://github.com/arbishakram/masked_regression_code/blob/master/images/Fig_2.png" /></p>


# Citation
If this work is useful for your research, please cite our [Paper](https://link.springer.com/article/10.1007%2Fs11263-019-01256-3):
```bash
@article{khan_mr_ijcv_2019,
author="Khan, Nazar
and Akram, Arbish
and Mahmood, Arif
and Ashraf, Sania
and Murtaza, Kashif",
journal="International Journal of Computer Vision",
pages = "1433--1454",
title = "{Masked Linear Regression for Learning Local Receptive Fields for Facial Expression Synthesis}",
volume = "128",
year = "2020"
}
```

