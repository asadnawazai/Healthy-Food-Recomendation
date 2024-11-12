from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load(r'C:\Users\Dell User\Desktop\HealthyFoodRecommendation\food_recommendation_model (1).pkl')
scaler = joblib.load(r'C:\Users\Dell User\Desktop\HealthyFoodRecommendation\scaler (1).pkl')
label_encoder_disease = joblib.load(r'C:\Users\Dell User\Desktop\HealthyFoodRecommendation\label_encoder_disease.pkl')
label_encoder_food = joblib.load(r'C:\Users\Dell User\Desktop\HealthyFoodRecommendation\label_encoder_food.pkl')
label_encoder_dutch = joblib.load(r'C:\Users\Dell User\Desktop\HealthyFoodRecommendation\label_encoder_dutch.pkl')
label_encoder_german = joblib.load(r'C:\Users\Dell User\Desktop\HealthyFoodRecommendation\label_encoder_german.pkl')

# Define the recommendation function
def recommend_foods(disease_input, top_n=5):
    # Encode the disease
    disease_encoded = label_encoder_disease.transform([disease_input])[0]
    
    # Prepare data for each category
    user_food_data = [[label_encoder_food.transform([food])[0], disease_encoded, 0, 0] for food in label_encoder_food.classes_]
    dutch_food_data = [[0, disease_encoded, label_encoder_dutch.transform([dish])[0], 0] for dish in label_encoder_dutch.classes_]
    german_food_data = [[0, disease_encoded, 0, label_encoder_german.transform([dish])[0]] for dish in label_encoder_german.classes_]
    
    # Scale the data
    user_food_data_scaled = scaler.transform(user_food_data)
    dutch_food_data_scaled = scaler.transform(dutch_food_data)
    german_food_data_scaled = scaler.transform(german_food_data)
    
    # Predict nutrients for each category
    user_predictions = model.predict(user_food_data_scaled)
    dutch_predictions = model.predict(dutch_food_data_scaled)
    german_predictions = model.predict(german_food_data_scaled)
    
    # Convert predictions to DataFrames and add food names
    user_recommendations = pd.DataFrame(user_predictions, columns=['Protein', 'Fats', 'Carbs', 'Calories'])
    user_recommendations['Food Name'] = label_encoder_food.inverse_transform([row[0] for row in user_food_data])
    user_recommendations = user_recommendations[['Food Name', 'Protein', 'Fats', 'Carbs', 'Calories']].round(1).sort_values(by='Calories').head(top_n)

    dutch_recommendations = pd.DataFrame(dutch_predictions, columns=['Protein', 'Fats', 'Carbs', 'Calories'])
    dutch_recommendations['Dutch Dish'] = label_encoder_dutch.inverse_transform([row[2] for row in dutch_food_data])
    dutch_recommendations = dutch_recommendations[['Dutch Dish', 'Protein', 'Fats', 'Carbs', 'Calories']].round(1).sort_values(by='Calories').head(top_n)

    german_recommendations = pd.DataFrame(german_predictions, columns=['Protein', 'Fats', 'Carbs', 'Calories'])
    german_recommendations['German Dish'] = label_encoder_german.inverse_transform([row[3] for row in german_food_data])
    german_recommendations = german_recommendations[['German Dish', 'Protein', 'Fats', 'Carbs', 'Calories']].round(1).sort_values(by='Calories').head(top_n)

    return user_recommendations, dutch_recommendations, german_recommendations

# Define routes
@app.route('/')
def form():
    return render_template('form.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    disease = request.form['disease']
    user_recs, dutch_recs, german_recs = recommend_foods(disease)
    return render_template(
        'form.html', 
        user_recs=user_recs.to_html(classes='data', header="true"),
        dutch_recs=dutch_recs.to_html(classes='data', header="true"),
        german_recs=german_recs.to_html(classes='data', header="true")
    )

if __name__ == "__main__":
    app.run(debug=True)