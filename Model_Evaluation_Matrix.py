#1. Accuracy (Overall Correctness)
#Logic: Kitni total predictions mein se kitni sahi thi?
#Formula: $\frac{TP + TN}{Total}$
#Problem: Agar aapke paas 100 log hain aur 99 healthy hain, 1 ko cancer hai. Model sab ko "Healthy" keh de toh accuracy 99% hogi, lekin model ne cancer patient ko miss kar diya. Isliye accuracy har jagah kaam nahi aati.

from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix,roc_auc_score,roc_curve
# Farz karein:
y_true = [0, 1, 0, 0, 1, 1, 0, 1] # Asli data
y_pred = [0, 1, 0, 0, 0, 1, 1, 1] # Model ki prediction

print(f"Accuracy is:{accuracy_score(y_true,y_pred)}")

#Precision (Quality over Quantity)
#Logic: "Jinko model ne Positive kaha, un mein se sach mein kitne Positive nikle?"
#Focus: False Positives (FP) ko kam karna.
#Use Case: Spam detection (Hum nahi chahte ke koi important mail ghalti se spam mein chali jaye).

print(f"Precision Score is:{precision_score(y_true,y_pred)}")

#Recall (Sensitivity / Detection Power)
#Logic: "Jitne sach mein Positive cases thay, model ne un mein se kitne pakre?"
#Focus: False Negatives (FN) ko kam karna.
#Use Case: Cancer detection (Hum ek bhi patient miss nahi karna chahte, bhale hi kisi healthy ko dubara test karna par jaye).

print(f"Recall is:{recall_score(y_true,y_pred)}")

## Confusion Matrix
print(f"Confusion Matrix is:{confusion_matrix(y_true,y_pred)}")


#ROC-AUC (The Probability Benchmark)
#ROC Curve: Ye ek graph hai jo "True Positive Rate" aur "False Positive Rate" ke darmiyan hota hai different thresholds par.
#AUC (Area Under Curve): Ye curve ke niche ka area hota hai.
#1.0: Perfect model.
#0.5: Be-kaar model (Tukka/Random guessing).

# AUC Score
auc = roc_auc_score(y_true, y_pred)
print(f"AUC Score: {auc}")
import matplotlib.pyplot as plt

# Curve Plotting
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()