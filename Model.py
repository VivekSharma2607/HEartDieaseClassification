#Split data into x and y
x = df.drop("target" , axis = 1)
y = df["target"]
np.random.seed(42)
x_train , x_test , y_train , y_test = test_train_split(x,y,test_size = 0.2 , random_state = 0)
#Putting model in Dictionary
models = {"Logistic Regression" : LogisticRegression(),
        "KNN" : KNeighborsClassifier(),
        "Random Forest Classifier" : RandomForestClassifier()}
#Using a utility function which will traverse through 3 of the algorithm and give us the prediction score
def check(models, X_train, X_test, y_train, y_test):
    """
    This function fits all the model type assigned above then loop through them and tells the best model to use
    x_train = training data (no labels)
    x_test = testing data (no labels)
    y_train = training labels
    y_test = test labels
    """
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
#Calling the function
m_score = check(models = models , x_train = x_train ,x_test = x_test, y_train = y_train , y_test = y_test)
m_score
#Graph for camparing model
m_campare = pd.DataFrame(m_score , index = ["Accuracy"])
m_campare.T.plot.bar();
