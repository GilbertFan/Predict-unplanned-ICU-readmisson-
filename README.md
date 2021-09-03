# Predict unplanned ICU readmisson
Outcome-Oriented Predictive Process Monitoring to Predict Unplanned ICU Readmission in MIMIC-IV Database\
Requirements:\
python==3.7\
pandas==0.25.3\
numpy==1.17.3\
Keras==2.3.1\
tensorflow==2.0.0

Creating a folder named data, and put testing and training data inside. The implementations current only support data with csv format.\
Run 'python p2.py' first, 'max_test_len' is the one to modify to test different prefix lengths.\
'min_len' and 'max_len' control the length of selected trace.\
Run 'python p3.py'\
Then run different implementations: e.g. 'python lstm.py'


