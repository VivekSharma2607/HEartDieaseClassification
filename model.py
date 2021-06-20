#Split data into x and y
x = df.drop("target" , axis = 1)
y = df["target"]
np.random.seed(42)
x_train , x_test , y_train , y_test = test_train_split(x,y,test_size = 0.2 , random_state = 0)
#Using a utility function which will traverse through 3 of the algorithm and give us the prediction score
def fit_and_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
#Calling the function
m_score = fit_and_score(models = models , x_train = x_train ,x_test = x_test, y_train = y_train , y_test = y_test)
m_score
#Graph for camparing model
m_campare = pd.DataFrame(m_score , index = ["Accuracy"])
m_campare.T.plot.bar();
