'''
Implement a Decision Tree classifier using the ID3 algorithm to segment customers based
on their purchasing behavior using the Online Retail dataset. Analyze the tree structure
and discuss the feature importance.
Tasks:
● Load and preprocess the Online Retail dataset.
● Implement Decision Tree using the ID3 algorithm.
● Visualize the decision tree and analyze feature importance.
● Discuss how the tree structure helps in understanding customer behavior.
'''

import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier,plot_tree

online_retail=fetch_ucirepo(id=352)
df=online_retail.data.original
print("original shape: ",df.shape)

df=df.dropna(subset=["CustomerID"])
df=df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df=df[(df["Quantity"]>0)&(df["UnitPrice"]>0)]
df["TotalAmount"]=df["Quantity"]*df["UnitPrice"]
print("After preprocessing: ",df.shape)

customer_data=df.groupby("CustomerID").agg(TotalSpent=("TotalAmount","sum"),TotalQuantity=("Quantity","sum"),
NumInvoices=("InvoiceNo","nunique"),UniqueProducts=("StockCode","nunique"),AvgOrderValue=("TotalAmount","mean")).reset_index()
customer_data["PurchaseFrequency"]=(customer_data["NumInvoices"]/customer_data["NumInvoices"].max())
customer_data["Segment"]=pd.qcut(customer_data["TotalSpent"],q=3,labels=["Low","Medium","High"])

print("Segment Contribution: ")
print(customer_data["Segment"].value_counts())
features=["TotalQuantity","NumInvoices","UniqueProducts","AvgOrderValue","PurchaseFrequency"]

X=customer_data[features]
y=customer_data["Segment"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

id3_model=DecisionTreeClassifier(criterion="entropy",max_depth=4,random_state=42)
id3_model.fit(X_train,y_train)
y_pred=id3_model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Results: ")
print("accuracy: ",accuracy)
print("Classification report: ")
print(classification_report(y_test,y_pred))

plt.figure(figsize=(20,10))
plot_tree(id3_model,feature_names=features,class_names=["Low","Medium","High"],rounded=True,fontsize=10,impurity=False)
plt.title("ID3 deecision tree for customer segmentation")
plt.show()

importance=pd.DataFrame({"Feature":features,"Importance":id3_model.feature_importances_})
importance=importance.sort_values(by="Importance",ascending=False)
print("Feature importance")
print(importance)

plt.figure(figsize=(8,5))
plt.bar(importance["Feature"],importance["Importance"])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature importance- ID3 decision tree")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()