# NLP-project-using-wine-reviews-for-origin-country-prediction

A classical text classification task in Natural Language Processing. 

The project aims at predicting the origin country of the using the provided description. The deep learning model applied in this project is  convolutional neural network model with word embeddings on Tensorflow. In this project, I used the dataset containing over 130k records for wine reviews on Kaggle. The best model accuracy was 0.716, and when applying the model to the 100 testing descriptions, 76 of them were given the right predictions.

An example of prediction:

    Description:

    Elegance, complexity and structure come together in this drop-dead gorgeous winethat ranks among Italy's greatest whites. It opens with sublime yellow spring flower, aromatic herb and orchard fruit scents. The creamy, delicious palate seamlessly combines juicy white peach, ripe pear and citrus flavors while white almond and savory mineral notes grace the lingering finish.

The ideal prediction of the above description should be: `Italy`.

This is a multi-class text classification task. In total, we have got 48 countries (labels) as the prediction results.

## Methodology

**NOTICE: Depending your data set, you should make your own choice over using whether CPU or GPU. Make sure you change the configuration of device in [text_cnn_rnn.py](https://github.com/youko70s/Using-Wine-Reviews-for-Country-Origin-Prediction/blob/master/text_cnn_rnn.py)**.

In NLP tasks, the input are sentences or documents represented as matrix. In this specific project, our input is the description which includes several sentences. So how it refers to a matrix? Each row of the matrix corresponds to one token, typically a word. Each row is a vector representing a word. Both word embeddings such as word2vec and one-hot vectors can be applied to represent those vectors. The filters in a CNN can be used to slide over full rows of the matrix. A typical example for CNN on NLP tasks will be like the following:

![cnn_model](/images/cnn_model.png)

## Data

Data set downloaded from Kaggle. Link: https://www.kaggle.com/zynicide/wine-reviews

In this project, we only use columns `description` and `country`. However, the work can be extended using other columns. 

## How to run

### preprocessing and preparing

*  [data_helper.py](https://github.com/youko70s/Using-Wine-Reviews-for-Country-Origin-Prediction/blob/master/data_helper.py): preprocessing and helper function. Preprocessing mainly includes lowercasing the text input and do lemmatizing; helper functions include creating padding, building inverted index for vocabulary, etc.

* [text_cnn_rnn.py](https://github.com/youko70s/Using-Wine-Reviews-for-Country-Origin-Prediction/blob/master/text_cnn_rnn.py): stores the CNN model.

* [training_config.json](https://github.com/youko70s/Using-Wine-Reviews-for-Country-Origin-Prediction/blob/master/training_config.json): the configuration of the deep learning model.


### Training phase

Refer to: [train.py](https://github.com/youko70s/Using-Wine-Reviews-for-Country-Origin-Prediction/blob/master/train.py).

    $ python train.py [train_file_path]

`[train_file_path]`: the path of the training dataset. Considering the size of data set, please use zip file. However, you can make your own changes by making changes to function `load_data()` in `data_helper.py`.

### Predicting phase

Refer to: [predict.py](https://github.com/youko70s/Using-Wine-Reviews-for-Country-Origin-Prediction/blob/master/predict.py).

During the training phase, a directory was created, and the trained model was automatically saved to that directory `./trained_results_[timecreated]`

    $ python predict.py [train_dir] [test_file_path]

`[train_dir]`: the path of the directory created at the training phase
`[test_file_path]`: the path to the test file dataset. In this case, I created `./data/small_sample.csv` for testing. You can also create your own test data.


The input of the testing data was generated as json file including only `country` and `description` as for better implementation of fitting the model. The output file was be stored as a csv file to `./predicted_results_[timecreated]`. 
 
`country`: representing the right label

`description`: representing the text input

`new_prediction`: representing the predicted result given by the CNN model

## Paper 

I also wrote a paper for this project and provided more detailed analysis and discussion. If you are interested in assessing the paper, or want to discuss over anything with me, feel free to email me: [youko1970s@gmail.com](mailto:youko1970s@gmail.com).




