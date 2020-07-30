# Semantically-Sensible-Thesaurus
This project is a django based app which provides possible word replacements for the detected verb in an input sentence using three different models. In order to run the app on the local machine, run the following command on the command line:  
$python manage.py runserver$  
The main project functions are present in views.py in SST. When you input a sentence, you can choose any of the three models- Tensor, BERT, and GOod English. After choosing, the word replaceemnts are provided in the next screen along with a basic algorithmic process used by that model. The good english model allows the possibility of online learning, i.e., if the user is not satisfied with the answers given by the model, he/she may enter better alternatives whihc eill be learned by the model on-the-go. The function for the same can be found in _online learning_ in views.py.  
This app is not going to run as it is because it requires 3 additional files- coretensor.npy, latentfactors.npy, and score_object_simple. coretensor.npy and latentfactors.npy are used by Tensor model and score_object_simple is used by Good English model.  

The Supplementary Material folder contains a Report.pdf which contains details of the entire project. Please refere to it for the same. It also contains the models files, each of whiich contains model training and evaluation. The code to make coretensor.npy, latentfactors.npy, and score_object_simple can be found in these files. The training and evaluation datasets used can be found in the report. The evaluation is done using spearman's correlation metric, details of which can be found in the report and code can be found in these supplementary files.

## Contributors
[Kanishk Kalra](https://github.com/kanishkkalra11)  
[Shweta Pardeshi](https://github.com/shwetapardeshi1)  
[Rohan Patil](https://github.com/bridgesign)  
[Kavita Vaishnaw](https://github.com/kavita-v)
