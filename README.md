DeepFake Video Detection 
—----------------------------------------------------------------------------------------------------------------------------


 A system which provides a mechanism for detecting whether the input video is real or fake by frame-by-frame processing of the video utilizing face detection and face classification algorithms


Input : Video


Output : Prediction whether the given video is real or synthesized


–----------------------------------------------------------------------------------------------------------------------------


URL/Source dataset links :


1. The following link contains the sample DeepFake Detection Challenge (DFDC) dataset provided by Kaggle which was used to train and test the model


            https://www.kaggle.com/competitions/deepfake-detection-challenge/data


           The dataset contains :
           400 train videos
           400 test videos
           Metadata file - Video reference ID , label


2. The following link contains the Face Forensics++ (FF++)  dataset provided by Kaggle which was used for training the XceptionNet model to improve the prediction accuracy


           https://www.kaggle.com/datasets/sorokin/faceforensics


         
—----------------------------------------------------------------------------------------------------------------------------


Implementation :


Hardware :


* System processor : intel core i3, i5, i7
* RAM              : Min 4GB
* Hard disk space  : 200GB or above


Software :


* Programming language : Python
* OS                   : Windows version 10, Linux
* Framework            : Keras, Tensorflow
* Tools                : Google colab
* GPU                  :  NVIDIA Tesla K80 GPU equipped with a 12GB PCIe slot available in Google Colab


—----------------------------------------------------------------------------------------------------------------------------




Instructions to execute the source code(COLAB) :


1. Store the input video in drive
2. To install the dependencies and necessary libraries, the initial requirement cells are executed
3. Specify the input drive path and mount the drive to google colab using mount-drive cell
4. The frame split-up module code is run and the extracted frames are stored in the storage path given in drive
5. Run the keyframe extraction module cells to identify the keyframes and to delete the other frames
6. Load the mobileNetv2 face extractor and run the face detection code to apply bounding boxes to the detected faces
7. Run the face extraction code to extract the bounding box from the remaining image and to store it separately
8. The faces are stored under the folder named after the video in the drive
9. Load the XceptionNet face classifier model and run the prediction cells to get the output of the video as REAL/FAKE.



Instructions to execute the source code(VS Code) :

The Local Execution may Requires GPU support for Faster Computation

1. clone the repositary in your local
2. install all the dependencies on your system
3. mention the path of the saved model
4. give the right path to your input source video and other directories
5. run the DeepfakeDetection.py in your system 
6. here you can download the trained model for face detection(MobileNEtV2) - https://drive.google.com/file/d/1i0ksAnGZ_LT2xfyhYRxnObLAZiBUhiH6/view?usp=drive_link
7. here you can download the trained mode for output classification(XceptionNet) - https://drive.google.com/file/d/1AB-iLvPkKiw3BehaCa4zDBH8qWYdY6L8/view?usp=drive_link




—----------------------------------------------------------------------------------------------------------------------------
