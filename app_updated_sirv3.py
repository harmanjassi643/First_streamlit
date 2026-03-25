# Basic Structure

import streamlit as st

import pandas as pd

from joblib import load

# load model and columns
model = load('tree.joblib')

train_cols = load('columns.joblib')

data = load('data.joblib')

st.set_page_config(page_title = "🚢 Titanic Dashboard", layout = "wide")

st.title("🚢 Titanic Dashboard")

st.markdown("*Interactive ML Dashboard with Titanic UI*")

st.divider()

st.sidebar.header("🎯 Passenger Details")
# sidebar
Pclass = st.sidebar.radio("Passenger Class", [1,2,3])

Sex = st.sidebar.radio("Sex", ["Male","Female"])
Sex = 1 if Sex == "Male" else 0

Embarked = st.sidebar.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"], key="embarked")
emb_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
Embarked = emb_map[Embarked]

Age_cat = st.sidebar.slider("Age Category (0:Child ,1:Teen ,2:Adult, 3:Senior)" , 0, 3, 1)
Fare_cat = st.sidebar.slider("Fare Category (0:Low ,1:Mid ,2:High ,3:Expensive)", 0, 3, 1)

Family = st.sidebar.radio("Family Onboard/?", ["No", "Yes"])
Family = 1 if Family == "Yes" else 0

input_df = pd.DataFrame({
    
    'Pclass': [Pclass],
    
    'Sex': [Sex],
    
    'Embarked': [Embarked],
    
    'Age Category': [Age_cat],
    
    'Fare Category': [Fare_cat],
    
    'Family': [Family]
})

input_df = input_df.reindex(columns = train_cols , fill_value = 0)

st.subheader("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Passenger" ,len(data))
    st.bar_chart(data['Pclass'].value_counts())

with col2:
    male_count = int((data['Sex'] == 1).sum())
    female_count = int((data['Sex'] == 0).sum())

    m1 , m2 = st.columns(2)

    with m1: 
        st.metric("Female", female_count)

    with m2:
        st.metric("Male", male_count)

    st.bar_chart(data['Sex'].value_counts())   

with col3:
    st.write("Unique Embarked", data['Embarked'].nunique())
    st.bar_chart(data['Embarked'].value_counts())

st.divider()

st.subheader("🔮 Prediction Panel")

col_pred, col_prob = st.columns(2)

with col_pred:
    if st.button("Predict"):
         result = model.predict(input_df)
         prob = model.predict_proba(input_df)

    

        if result[0] == 1:
                  st.success("Survived ✅")

        
    
    else:
        
        st.error("Did Not Survive ❌")

    st.session_state['prob'] = prob

with col_pred:  
        if 'prob' in st.session_state:
                     st.metric("Survival Probability", f"{round(st.session_state['prob'][0][1]*100,2)}%")



if st.button("Reset"):
        st.session_state.clear()
        st.rerun()


        st.divider()

st.subheader("🗃️ Input Summary")

st.write(input_df)