import os

sb_ratio = [0.001 , 0.00215443, 0.00464159,0.01, 0.02154435, 0.04641589, 0.1 , 0.21544347  ,0.46415888 , 1.]

os.mkdir('../Results/Classifier')
os.mkdir('../Results/Classifier/sbvsb')
os.mkdir('../Results/Classifier/sbvsb/tpr')
os.mkdir('../Results/Classifier/sbvsb/fpr')
os.mkdir('../Results/Classifier/sbvsb/auc')
for i in sb_ratio:
    os.mkdir('../Results/Classifier/sbvsb/tpr/'+str(i))
    os.mkdir('../Results/Classifier/sbvsb/fpr/'+str(i))
    os.mkdir('../Results/Classifier/sbvsb/auc/'+str(i))
