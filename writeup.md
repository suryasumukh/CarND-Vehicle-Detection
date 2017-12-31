# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_vehicles.png
[image2]: ./output_images/hog_non_vehicles.png
[image3]: ./output_images/scale1.png
[image4]: ./output_images/scale2.png
[image5]: ./output_images/scale3.png
[image6]: ./output_images/pipeline_test.png
[image7]: ./output_images/test_image1.png
[image8]: ./output_images/test_image2.png
[image9]: ./output_images/test_image3.png
[image10]: ./output_images/test_image4.png
[image11]: ./output_images/test_image5.png
[image12]: ./output_images/test_image6.png


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cells 5 & 6 of the IPython notebook `notebooks/detect-vehicles.ipynb`. Two functions `feature_extractor()` and `feature_extractor_w_hist()` extract HOG features from vehicle and non-vehicle images. A wrapper function called `read()` accepts a directory path and a function which extractors features from images as arguments and iterates through all images in the directory and applies the function to the image to extract HOG features and other features for training a classifier. `feature_extractor()` and `feature_extractor_w_hist()` use `skimage.hog()` to get HOG features. The color space and HOG parameters tuning are explained in the next section.

```python
def hog_feature(image, orientations=9):
    return hog(image, orientations=orientations,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               feature_vector=True)

def feature_extractor(image, orientations=9):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    # HOG Features
    rhf = hog_feature(img[:, :, 0], orientations)
    ghf = hog_feature(img[:, :, 1], orientations)
    bhf = hog_feature(img[:, :, 2], orientations)
    return np.hstack((rhf, ghf, bhf))

def feature_extractor_w_hist(image, orientations=9):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    # HOG Features
    rhf = hog_feature(img[:, :, 0], orientations)
    ghf = hog_feature(img[:, :, 1], orientations)
    bhf = hog_feature(img[:, :, 2], orientations)
    # Histogram Features
    channel1_hist = np.histogram(img[:,:,0], bins=32)
    channel2_hist = np.histogram(img[:,:,1], bins=32)
    channel3_hist = np.histogram(img[:,:,2], bins=32)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return np.hstack((rhf, ghf, bhf, hist_features))

def read(data_dir, fn):
    for img_path in glob.iglob(os.path.join(data_dir, '*', '*.png')):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        yield fn(image)
```

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**Vehicles**  
![alt text][image1]  
**Non Vehicles**    
![alt text][image2]    

#### 2. Explain how you settled on your final choice of HOG parameters.

The final parameters for HOG was chosen as part of GridSearchCV for SVM classifier. The HOG features were treated a hyper-params and a sklearn pipeline TransformationMixin was created so that it would be easier to implement GridSearchCV. The code can be found in cell 7 of `notebooks/classifier.ipynb`. The following were the grid params.

```python
color_schemes = [cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HLS, cv2.COLOR_BGR2YCrCb]
orientations = [9, 11, 13]
pixels_per_cell = [8, 16]
cells_per_block = [2, 3]
param_grid = dict(hog__color_scheme=color_schemes,
                  hog__orientation=orientations,
                  hog__pixels_per_cell=pixels_per_cell,
                  hog__cells_per_block=cells_per_block)
```
The results of GridSearchCV is explained in the next section. While GridSearchCV suggested the following params -
```
{'hog__cells_per_block': 2,
 'hog__color_scheme': 36, # YCrCb
 'hog__orientation': 13,
 'hog__pixels_per_cell': 8
}
```
It was found that HOG features with `orientations=13` produced a lot of False Positives, `orientations=9` was chosen instead and produced fewer False Positives. All channels of YCrCb was used to generate features for the classifier.
The final parameters were -
```
{'hog__cells_per_block': 2,
 'hog__color_scheme': 36, # YCrCb
 'hog__orientation': 9,
 'hog__pixels_per_cell': 8
}
```   

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A sklearn Pipeline was created with `StandardScalar()` transformation and `LinearSVC()` classifier. A temporary `HogFeatureExtractor()` transformer was created to aid HOG parameter search. GridSearchCV with `cv=3` was employed to find best performing classifier. The code can be found in cells 7-20 of `notebooks/classifier.ipynb`. The hyper-params mostly consisted of HOG params. The out of the box params for `LinearSVC()` were sufficient to get a high f1 score and did not need tuning. `C=1` was used for the SVM for a `max_iter=1000`.

GridSearchCV was performed on two sets of features, one consisting of only HOG features and the other consisting of histogram features in addition to HOG features. F1 score was chosen as the scoring function for grid search. The results of GridSearchCV

| config | cells_per_block | color_scheme | orientation | pixels_per_cell | f1_train_hog | f1_train_hog_w_hist | f1_test_hog | f1_test_hog_w_hist |
|--------|-----------------|--------------|-------------|-----------------|--------------|---------------------|-------------|--------------------|
| 0      | 2               | 4            | 9           | 8               | 1            | 1                   | 0.963889    | 0.976151           |
| 1      | 2               | 4            | 9           | 16              | 0.998134     | 0.99954             | 0.954056    | 0.96984            |
| 2      | 2               | 4            | 11          | 8               | 1            | 1                   | 0.966355    | 0.974732           |
| 3      | 2               | 4            | 11          | 16              | 0.999261     | 0.999717            | 0.955341    | 0.972904           |
| 4      | 2               | 4            | 13          | 8               | 1            | 1                   | 0.966127    | 0.976717           |
| 5      | 2               | 4            | 13          | 16              | 0.999648     | 0.999965            | 0.959135    | 0.974696           |
| 6      | 2               | 52           | 9           | 8               | 1            | 1                   | 0.982921    | 0.988252           |
| 7      | 2               | 52           | 9           | 16              | 0.999965     | 1                   | 0.971237    | 0.985893           |
| 8      | 2               | 52           | 11          | 8               | 1            | 1                   | 0.982438    | 0.988455           |
| 9      | 2               | 52           | 11          | 16              | 0.99993      | 1                   | 0.973128    | 0.987495           |
| 10     | 2               | 52           | 13          | 8               | 1            | 1                   | 0.984349    | 0.988813           |
| 11     | 2               | 52           | 13          | 16              | 0.999965     | 1                   | 0.974446    | 0.987364           |
| 12     | 2               | 36           | 9           | 8               | 1            | 1                   | 0.987161    | 0.991289           |
| 13     | 2               | 36           | 9           | 16              | 1            | 1                   | 0.984241    | 0.989394           |
| 14     | 2               | 36           | 11          | 8               | 1            | 1                   | 0.987645    | 0.990998           |
| 15     | 2               | 36           | 11          | 16              | 0.999965     | 1                   | 0.986141    | 0.990594           |
| 16     | 2               | 36           | 13          | 8               | 1            | 1                   | 0.988783    | 0.99199            |
| 17     | 2               | 36           | 13          | 16              | 1            | 1                   | 0.986857    | 0.990599           |
| 18     | 3               | 4            | 9           | 8               | 1            | 1                   | 0.963401    | 0.975405           |
| 19     | 3               | 4            | 9           | 16              | 0.988052     | 0.999328            | 0.948823    | 0.964556           |
| 20     | 3               | 4            | 11          | 8               | 1            | 1                   | 0.963909    | 0.975015           |
| 21     | 3               | 4            | 11          | 16              | 0.995328     | 0.999434            | 0.945584    | 0.965114           |
| 22     | 3               | 4            | 13          | 8               | 1            | 1                   | 0.963226    | 0.974705           |
| 23     | 3               | 4            | 13          | 16              | 0.999226     | 0.999788            | 0.952299    | 0.968561           |
| 24     | 3               | 52           | 9           | 8               | 1            | 1                   | 0.978552    | 0.987548           |
| 25     | 3               | 52           | 9           | 16              | 0.999753     | 0.999929            | 0.965851    | 0.982687           |
| 26     | 3               | 52           | 11          | 8               | 1            | 1                   | 0.979664    | 0.987478           |
| 27     | 3               | 52           | 11          | 16              | 0.999754     | 0.999965            | 0.968013    | 0.984698           |
| 28     | 3               | 52           | 13          | 8               | 1            | 1                   | 0.981402    | 0.987825           |
| 29     | 3               | 52           | 13          | 16              | 0.999965     | 0.999965            | 0.971001    | 0.985186           |
| 30     | 3               | 36           | 9           | 8               | 1            | 1                   | 0.987381    | 0.99058            |
| 31     | 3               | 36           | 9           | 16              | 0.99993      | 0.999965            | 0.979681    | 0.986369           |
| 32     | 3               | 36           | 11          | 8               | 1            | 1                   | 0.988571    | 0.990725           |
| 33     | 3               | 36           | 11          | 16              | 1            | 1                   | 0.982772    | 0.987289           |
| 34     | 3               | 36           | 13          | 8               | 1            | 1                   | 0.98788     | 0.99143            |
| 35     | 3               | 36           | 13          |                 | 0.99993      |                     | 0.983923    |                    |


All 3 color channels of YCrCb along with histogram features with bin size 32 were used as features. As explained earlier `orientations=13` produced more False Positives, so `orientations=9` were chosen. With the newly chosen parameter a separate pipeline was constructed without the `HogFeatureExtractor()` transformer was trained and the pipeline was saved in a pickle file. The following is the Accuracy, F1 Score and Confusion Matrix for the final params.

Final Params:  
```
{'hog__cells_per_block': 2,
 'hog__color_scheme': 36, # YCrCb
 'hog__orientation': 9,
 'hog__pixels_per_cell': 8
}
```

Accuracy: 0.98902027027    
F1 Score: 0.98872506505  
FP: 19  
FN: 20  
TP: 1803  
TN: 1710  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As suggested, HOG features were extracted once for the bottom half of the image and slices were used at prediction time. Three scales were chosen to convolve over the bottom half of the image. The following are the images of sliding window scales. The relevant code can be found in cells 13 & 22 of `notebooks/sliding-window.ipynb`.

A scale size of 1 and cell overlap of 1 was used between 400 and 500 pixels on the image.
![alt text][image3]  

A scale of size 2 and cell overlap of 2 was used between 400 and 600 pixels.  
![alt text][image4]  

The third scale with size 3 and cell overlap of 3 was used between 400 and 670 pixels.  
![alt text][image5]  

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The classifier was trained with the above described HOG params along with histogram features of all 3 channels of YCrCb color scheme. Sliding windows with scales defined above are used to detect potions of the image that contain images. The following it the output on test images.

![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_hist.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As suggested in the lesson, heatmap and threshold were used to reduce some of the false positives. A threshold of 1 is applied for the heatmap of every frame. `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and bounding boxes were drawn to reflect the blobs on the original image. For the video generation, a FIFO quue of size 5 was used to smooth out detections in the frame. If a vehicle was detected, the current and the pervious 5 heatmaps were summed and a threshold of 2 was applied. If no vehicle was detected only the previous two heatmaps were summed. The following are images from the pipeline. The relevant code can be found in cells 15-17 and 25 of `notebooks/sliding-window.ipynb`

Here are six frames and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The classifier does make false positives with heatmap filtering. The pipeline draws separate bounding boxes separated by a small number of pixels for same vehicle. A rule can be used to merge bounding boxes if they are separate by a small distance to make detection more robust. The model fails for oncoming traffic and cross traffic. There are some images of cross traffic in the dataset, but the numbers are skewed. The pipeline also fails to detect pedestrians, bi-cycles and building which might be important, not to mention bad lighting and weather conditions. This is because the model is heavily driven by HOG features. Convolutional Neural Networks could another class of models that are a better fit for such tasks.
