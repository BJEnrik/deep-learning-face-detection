# <center>Two-Stage Facial Recognition and Classification Model Using Multi-Task Cascaded Convolutional Neural Networks and Inception-ResNet v1</center>

---

# Executive Summary

This technical report outlines the creation of a two-stage model for:
1. Face Detection (Notebook 1)
2. Face Classification (Notebook 2)

The two-stage model utilized state-of-the-art deep learning and tools for both face detection and face classification tasks. For face detection, I used the `MTCNN` (Multi-Task Cascaded Convolutional Neural Networks), which is a deep learning algorithm built into the `facenet_pytorch`, an open-source facial recognition toolkit. For face classification, on the other hand, I used the `InceptionResNetV1` architecture, which was pre-trained on the `vggface2` dataset, a large-scale face recognition dataset. The two-stage model is capable of accurately extracting, cropping, and classifying each Capstone Team Member's face.

Getting to the results, the `face detector` was able to accurately detect faces with impressive prediction scores above **99%** especially for the Capstone Team Member's Faces.

Furthermore, the face classifier was trained for **50 epochs** and achieved an impressive test accuracy of **0.9643**. 

The model's success can be attributed to its use of powerful neural network architectures pre-trained on large datasets. 

This is part 1 of a 2-part Notebook entitled *Two-Stage Facial Recognition and Classification Model Using Multi-Task Cascaded Convolutional Neural Networks and Inception-ResNet v1*.

# References

[1] Serengil, S. (2020). Deep Face Detection with MTCNN in Python. Retrieved from https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/

[2] Chintaram, Y. (2021). Automating Attendance System using Face Recognition with Masks. Retrieved from https://github.com/yudhisteer/Face-Recognition-with-Masks

[3] Elgendy, M. (2020, November 10). Deep Learning for Vision Systems. Manning Publications Co.
*Transfer Learning for Computer Vision Tutorial — PyTorch Tutorials 2.0.0+cu117 documentation*. (n.d.). Transfer Learning for Computer Vision Tutorial — PyTorch Tutorials 2.0.0+cu117 Documentation. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

[4] Alzubaidi, L., Santamaría, J., Manoufali, M., Mohammed, B., Fadhel, M. A., Zhang, J., Al-Timemy, A.H., Al-Shamma, O. & Duan, Y. (2021). MedNet: pre-trained convolutional neural network model for the medical imaging tasks. *arXiv preprint* arXiv:2110.06512.

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).

[6] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4700-4708).

[7] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). Imagenet classification with deep convolutional neural networks. *Communications of the ACM*, 60(6), 84-90.

[8] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510-4520).

[9] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint* arXiv:1409.1556.

[10] Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In *International conference on machine learning* (pp. 6105-6114). PMLR.
