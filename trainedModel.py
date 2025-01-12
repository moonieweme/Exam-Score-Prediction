import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model("trainedStudentPreformanceModel.keras")

    
# 1-Low 2-Medium 3-High || Parental_Involvement, Access_to_Resources, Motivation_Level, Family_Income, Teacher_Quality
# 0-No 1-Yes || Extracurricular_Activities, Internet_Access, Learning_Disabilities
# 0-Public 1-Private || School_Type
# 0-Negative 1-Neutral 2-Positive || Peer_Influence
# Highschool-1 College-2 Postgraduate-3 || Parental_Education_Level
# Near-1 Moderate-2 Far-3 || Distance_from_Home
# Male-0 Female-1 || Gender

Hours_Studied = 25.0
Attendance = 90
Parental_Involvement = 3 # High
Access_to_Resources = 3 # High
Extracurricular_Activities = 1 # Yes
Sleep_Hours = 100
Previous_Scores = 90 # // Previous Score positive - positive
Motivation_Level = 1 # Low
Internet_Access = 1 # Yes  // Having internet access - positive
Tutoring_Sessions = 0
Family_Income = 2 # Medium
Teacher_Quality = 1 # low
School_Type = 1 # Private
Peer_Influence = 2 # Positive
Physical_Activity = 8
Learning_Disabilities = 0 # No
Parental_Education_Level = 3 # Postgraduate
Distance_from_Home = 1 # Near
Gender = 0 #Male

student_data = np.array([[
Hours_Studied,
Attendance,
Parental_Involvement,
Access_to_Resources,
Extracurricular_Activities,
Sleep_Hours,
Previous_Scores,
Motivation_Level,
Internet_Access,
Tutoring_Sessions,
Family_Income,
Teacher_Quality,    
School_Type,
Peer_Influence,
Physical_Activity,
Learning_Disabilities,
Parental_Education_Level,
Distance_from_Home,
Gender
]], dtype=np.float32)

print("The model predicts your student would score: ", int((loaded_model.predict(student_data))[0][0]))
