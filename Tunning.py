#Tunning using RandomizedSearchCV is in JupyterNotebook
#Tunning using GridSearchCV on Logistic Regressor and RandomForestClassifier 
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
#Creating Classification Report using Cross Validation Report
## Checkig the best parameter
gs_log_reg.best_params_
clf = LogisticRegression(C = 1.3738237958832638 , solver = "liblinear")  
# Calculating the accuracy
clf_acc = cross_val_score(clf , x , y , cv = 5 , scoring = "accuracy")
clf_acc = np.mean(clf_acc)
clf_acc
## Calculating the precision
clf_pre = cross_val_score(clf , x , y , cv = 5 , scoring="precision")
clf_pre = np.mean(clf_pre)
clf_pre
clf_recall = cross_val_score(clf , x , y , cv = 5 , scoring="recall")
clf_recall = np.mean(clf_recall)
clf_recall
clf_f1 = cross_val_score(clf , x , y , cv = 5 , scoring="f1")
clf_f1 = np.mean(clf_f1)
clf_f1
# Visualize the data in form of graph
cv_metrics = pd.DataFrame({"Accuracy" : clf_acc,
                          "Precision" : clf_pre,
                          "Recall" : clf_recall,
                          "F1-score" : clf_f1},
                         index = [0])
cv_metrics.T.plot.bar(title = "Cross Validation CLassification Matrix" ,
                      legend = False );
