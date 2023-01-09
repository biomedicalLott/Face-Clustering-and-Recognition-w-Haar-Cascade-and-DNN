# Face-Clustering-and-Recognition-w-Haar-Cascade-and-DNN
A project designed to understand face detection and categorization. 

<h2>How to run</h2>
To use make sure to unpack the DNN model in the rar files. Once that's done, run main to run either the face detection or clustering. 



<h2>Project Description</h2>
The first step within it was to become familiarized with the haar cascade profiles provided through OpenCV and take time to understand how to apply the classifiers to best approach a decent result. An ensemble classifier ended up being the best use of the haar cascade. Unfortunately it was still not achieving about a 91% accuracy during clustering although it handled faces very well during simple detection. 

So to find an alternative, I turned to a deep neural network model trained for face recognition and implemented it within the project. At present the DNN is uncommented and the Haar cascade remains commented. 

The clustering was performed on this folder of images 
<p><img src="https://i.imgur.com/DNxOT6O.png"/></p>

resulting in the following clusters  with the DNN 
<p><img src="https://i.imgur.com/vJAFFZh.png"/></p>

