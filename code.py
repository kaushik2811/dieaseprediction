from tkinter import *
from tkinter.filedialog import asksaveasfile 
import numpy as np
import pandas as pd

# from gui_stuff import *
l1=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','chills','joint_pain','stomach_pain','muscle_wasting',
    'vomiting','burning_micturition','fatigue','lethargy','high_fever','sunken_eyes','breathlessness','sweating',
	'indigestion','headache','yellowish_skin','nausea','loss_of_appetite','diarrhoea','mild_fever',
	'acute_liver_failure','malaise','chest_pain','neck_pain','dizziness','cramps','muscle_weakness',
	'stiff_neck','loss_of_balance','weakness_of_one_body_side','bladder_discomfort','increased_appetite','pus_filled_pimples',
	'skin_peeling','blister']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv('training.csv')

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)


# ------------------------------------------------------------------------------------------------------
#spliting training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.34)
#-------------------------------------------------------------------------------------------
def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    temp = 0
    for symptom in psymptoms:
        if(symptom=="None"):
            temp+=1
    if(temp==5):
        t1.delete("1.0", END)
        t1.insert(END, "No symptoms entered")
        return 0;
    for k in range(len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]
    print("predicted = ", predicted)

    for a in range(len(disease)):
        if(predicted == a):
            t1.delete("1.0", END)
            t1.insert(END, disease[a])
            return
    t1.delete("1.0", END)
    t1.insert(END, "Not Found")
# gui_stuff------------------------------------------------------------------------------------

root = Tk()
root.configure(background='#383838')
root.title('Disease Prediction')

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

#Heading
w2 = Label(root, justify=LEFT, text="Disease Prediction using Machine Learning", fg="white", bg="#383838")
w2.config(font=("Elephant", 20))
w2.grid(row=1, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name of the Patient", fg="white", bg="#383838")
   
NameLb.grid(row=6, column=0, pady=15, sticky=W)


S1Lb = Label(root, text="Symptom 1", fg="white", bg="#383838")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="white", bg="#383838")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="white", bg="#383838")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="white", bg="#383838")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="white", bg="#383838")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="DecisionTree", fg="white", bg="#383838")
lrLb.grid(row=15, column=0, pady=10,sticky=W)



# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)
S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

#button
dst = Button(root, text="DecisionTree", command=DecisionTree,bg="green",fg="yellow")
dst.grid(row=8, column=3,padx=10)

#textfileds
t1 = Text(root, height=1, width=40,bg="orange",fg="black")
t1.grid(row=15, column=1, padx=10)


#saving result
# def save(): 
   # f=asksaveasfile(mode='w',defaultextension=".txt")
   # if f is None:
       # return
   # f.write("Name of the patient ")
   # f.write(NameEn.get())
   # f.write("First Symptom")
   # f.write(Symptom1.get())
   # f.write("Second Symptom")
   # f.write(Symptom2.get())
   # f.write("Third Symptom")
   # f.write(Symptom3.get())
   # f.write("Fourth Symptom")
   # f.write(Symptom4.get())
   # f.write("Fifth Symptom ")
   # f.write(Symptom5.get())
   # f.write("Predicted disease")

   # f.close()
              
# btn = Button(root, text = 'Save', command = lambda : save()) 
# btn.grid(row=11, column=3,padx=10)
root.mainloop()
