

```python
import time #timing measurements 
import numpy as np #data manipulation
from pandas import read_csv #reading data from a CSV file

#import random number generation
from random import seed #Sets a starting point for random number generation
from random import randint #Generates a random integer within a specified range

#Data Preparation Faces (list of labels for facial images)
ListofLabels = ['Akshay Kumar, Alexandra Daddario', 'Alia Bhatt, Amitabh Bachchan', ...]


#create a copy of ListofLabels and store it in SelectedFaces to keep track of which faces were chosen randomly
SelectedFaces = ListofLables.copy()
dataset = read_csv("/GP/Datasetfaces.csv")

Faces = dataset.iloc[:,0].values #Extracts image paths from the first column of the CSV and stores them in Faces
Lables = dataset.iloc[:,1].values #Extracts labels (corresponding to the images) from the second column of the CSV and stores them in Lables

seed(int(time.time())) #Sets the random number seed based on the current time, ensuring different random selections on different runs.


for i in range(len(ListofLabels)): #loop through each label (ListofLabels[i]) The goal is to randomly select one face for each label
  #Filters the Faces list based on the current label keeping only faces associated with that label.
  #Slices the filtered faces list to ensure it's the same length as the original Faces list (avoiding potential index errors)
  faces = (Faces[Labels == ListofLabels[i]])[:len(Faces)] 
  value = randint(0, len(faces)-1) #Generates a random integer within the range of available face indices (0 to len(faces)-1)
  SelectedFaces[i] = faces[value] #Updates the corresponding entry in SelectedFaces with the randomly chosen face path from the filtered faces list

print(SelectedFaces) #contains one randomly chosen face path for each label

#Saves the SelectedFaces list as (.npy file) This file will be used later in the enrollment and identification process.
filename = 'GP/selectedfaces.npy'
np.save(filename, SelectedFaces)


#Data preparation - fingerprints

#Read fingerprint data (paths or features) from a CSV file using pandas
dataset = read_csv("GP/Datasetfingerprints.csv")
Fingerprints = dataset.iloc[:,0].values #Extract fingerprints and labels from separate columns
Labels = dataset.iloc[:,1].values
print(len(Fingerprints))

seed(int(time.time()))

print(ListofLabels)

SelectedFingerprints = ListofLabels.copy() #Create a copy of ListofLabels for SelectedFingerprints


#loop through each label, filter fingerprints based on the label, randomly select one fingerprint per label, and update the SelectedFingerprints list
for i in range(len(ListofLabels)):
  fingerprints = (Fingerprints[Labels == ListofLabels[i]])[:len(Fingerprints)]
  value = randint(0, len(fingerprints)-1)
  SelectedFingerprints[i] = fingerprints[value]

#Print the final list of SelectedFingerprints
print(SelectedFingerprints)
filename = 'GP/selectedfingerprints.npy' #Save the SelectedFingerprints list as a NumPy array file "GP/selectedfingerprints.npy".
np.save(filename, SelectedFingerprints)
```
