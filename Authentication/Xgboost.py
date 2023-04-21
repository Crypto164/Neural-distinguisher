import numpy
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data

n_inputs = 630
n_classes = 4



csv_x_data = []
csv_tr_x_data = []
csv_tr_y_data = []
csv_xg_tr_y_data = []
csv_te_x_data = []
csv_te_y_data = []
csv_x_file = numpy.loadtxt("csv_protocol.csv", dtype=numpy.str, delimiter=',')
csv_xg_y_file = numpy.loadtxt("csv_xgboost_tag.csv", dtype=numpy.str, delimiter=',')

zero_row = numpy.zeros((1, n_inputs), dtype=numpy.float32)
	
row_count = 0
index = 0

for row in csv_x_file:
    if row[0] != '':
        row_count += 1
      
       
        csv_x_data.append(row.astype(numpy.float32))
        

    else:
        if row_count < 14:
            for iter in range(14 - row_count):
                csv_x_data.append(zero_row[0])
        csv_x_data = numpy.array(csv_x_data)
        
        csv_x_data = csv_x_data.reshape(-1, 14 * n_inputs)	
        csv_tr_x_data.append(csv_x_data)

        csv_x_data = []
        row_count = 0
        index += 1

csv_tr_x_data = numpy.array(csv_tr_x_data).reshape(547,8820)


for row in csv_xg_y_file:
        if row[0] != '':
            # if index % 5 == 0:
            #     csv_te_y_data.append(row.astype(numpy.float32) - 1)
            # else:
            #     csv_tr_y_data.append(row.astype(numpy.float32) - 1)

            csv_xg_tr_y_data.append(row.astype(numpy.int32) - 1)

            index += 1

Y = numpy.array(csv_xg_tr_y_data)


X=numpy.array(csv_tr_x_data)
print(X[0])

seed = 10
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)
# fit model on training data


param_dist = {'max_depth':30 , 'n_estimators': 30 }
model = XGBClassifier(**param_dist)
model.fit(X_train, y_train)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
