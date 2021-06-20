#Hyperparameter tunning using RandomizedSearchCV
#Logistic Regression()
#RandomForestClassifier()
#Tunning with GridSearch
# Diffrent hyperprameter for our LogisticREgrssion model As it was the best
log_reg_grid = {"C" : np.logspace(-4 , 4 , 30) ,
               "solver" : ["liblinear"]}
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv=5,
                         verbose=True)
#Fit the model
gs_log_reg.fit(x_train , y_train);
#Checing for the best parameter 
gs_log_reg.best_params_
gs_log_reg.score(x_test , y_test) #score
#Plotting ROC Curve
y_preds = gs_log_reg.predict(x_test)
plot_roc_curve(gs_log_reg , x_test , y_test);
#Confusion Matrix using a utility function
sns.set(font_scale=1.5)
def abc(y_test , y_preds):
    """
    Plotting the prediction on confusion matrix using the Seaborn
    """
    fig , ax = plt.subplots(figsize = (3,3))
    ax = sns.heatmap(confusion_matrix(y_test , y_preds),
                    annot=True,
                    cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicte Label")
abc(y_test , y_preds)
#Classification Report
print(classification_report(y_test , y_preds))
