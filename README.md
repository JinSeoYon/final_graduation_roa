# final_graduation

## Sookmyung Women's University

## Real-time Baby Cry Classification Application based On Artificial intelligence

Understanding crying, the only communication method of a baby who cannot speak, is the most difficult and difficult part of parenting. In particular, in the case of parents who are not familiar with child-rearing, and parents with hearing impairments, it is more difficult because they do not know the reason for the baby's crying. In this study, we propose a mobile application that uses the child's crying sound database to determine the child's crying sound in real time. By using the fact that the frequency of the crying sound varies depending on the child's condition, the sound is sensed in real time, and when the crying sound is heard, the child's crying sound is analyzed with a deep learning model to inform the reason for the child's crying sound. Considering that the situation that requires this service may be urgent, the user interface is simple and intuitively configured.




### Procedure

find your local ip address. 


## Run experiments
```
python roa_local.py
```

### if you want to train your own dataset

```
python roa_train.py
```


## Datasets
The datasets for ROA. donate-a-cry-corpus

Since the number of data in the donate-a-cry-corpus is not enough to train the deep learning model, a task of increasing the learning data was performed through data augmentation.

It can be downloaded at : hhttps://github.com/gveres/donateacry-corpus



## Acknowledgements and References

Kheddache,Y, Tadj C. “Frequential characterization of healthy and pathological newborn cries”, Am J Biomed Eng, 3 pp.182-193, 2013

 Krizhevsky, A., Sutskever, I., and Hinton, G. “Imagenet classification with deep convolutional neural networks,” Advances in Neural Information Processing Systems, pp. 1106–1114, 2012.

 J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. “DeCAF: A deep convolutional activation feature for generic visual recognition,” Proc. of International Conference on Machine Learning, pp. 647–655, 2013 [4] J. Salamon, J.P.Bello, “Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification”, Signal Process. Lett(SPL) 24 (3) pp. 279-283, 2017 

Koustav Chakraborty, Asmita Talele and Savitha Upadhya, "Voice Recognition Using MFCC Algorithm", International Journal of Innovative Research in Advanced Engineering (IJIRAE), vol. 1, no. 10, 2014.
