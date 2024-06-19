from flask import Flask, render_template, request, jsonify
import pandas as pd
import openai
import spacy

# Load NLP model
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

# Load the DataFrame from the pickle file
df = pd.read_pickle('recipes_data.pkl')

# Initialize OpenAI API key
openai.api_key = 'sk-proj-ny4vZT1hB8KbXL4tuctVT3BlbkFJYdtzNd2n7XjU0y6BYhs9'  # Replace this with your new API key


def generate_new_recipe(ingredients):
    prompt = f"Create a new and tasty recipe using the following ingredients: {', '.join(ingredients)}. Please provide a recipe name, ingredients list, and instructions."
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the correct model name
        prompt=prompt,
        max_tokens=150
    )
    generated_recipe = response.choices[0].text.strip()
    
    parts = generated_recipe.split('\n')
    recipe_name = parts[0]
    ingredients_list = "\n".join(parts[1:parts.index('Instructions:')])
    instructions = "\n".join(parts[parts.index('Instructions:')+1:])
    
    return recipe_name, ingredients_list, instructions

def find_matching_recipe_by_name(recipe_name):
    filtered_recipes = df[df['RecipeName'].str.lower().str.contains(recipe_name.lower())]
    if not filtered_recipes.empty:
        recipe = filtered_recipes.iloc[0]
        return recipe['RecipeName'], recipe['Ingredients'], recipe['Instructions'], int(recipe['TotalTimeInMins'])
    return None, None, None, None

def find_matching_recipe(ingredients, cuisine):
    filtered_recipes = df.copy()
    if ingredients:
        for ingredient in ingredients:
            filtered_recipes = filtered_recipes[filtered_recipes['Ingredients'].str.lower().str.contains(ingredient.strip().lower(), na=False)]
    if cuisine:
        filtered_recipes = filtered_recipes[filtered_recipes['Cuisine'].str.lower() == cuisine.lower()]
    if not filtered_recipes.empty:
        recipe = filtered_recipes.sample(1)
        return recipe['RecipeName'].iloc[0], recipe['Ingredients'].iloc[0], recipe['Instructions'].iloc[0], int(recipe['TotalTimeInMins'].iloc[0])
    return None, None, None, None

def process_user_input(user_input):
    doc = nlp(user_input)
    ingredients = [ent.text for ent in doc.ents if ent.label_ == 'FOOD']
    if not ingredients:
        ingredients = [token.text for token in doc if token.pos_ == 'NOUN']
    
    recipe_name = None
    for token in doc:
        if token.dep_ == 'dobj' and token.head.lemma_ == 'give':
            recipe_name = token.text

    if recipe_name:
        recipe_name, matched_ingredients, instructions, cooking_time = find_matching_recipe_by_name(recipe_name)
    elif ingredients:
        recipe_name, matched_ingredients, instructions, cooking_time = find_matching_recipe(ingredients, '')
        if not recipe_name:
            recipe_name, matched_ingredients, instructions = generate_new_recipe(ingredients)
            cooking_time = "Unknown"
    else:
        recipe_name, matched_ingredients, instructions, cooking_time = None, None, None, None
    
    return recipe_name, matched_ingredients, instructions, cooking_time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message'].strip()
    
    # Handle simple greetings
    if user_input in ['hi', 'hello', 'hey']:
        response = {"message": "Hello! How can I assist you today? You can ask for a recipe by name or provide ingredients and optionally a cuisine type."}
        return jsonify(response)
    
    recipe_name, matched_ingredients, instructions, cooking_time = process_user_input(user_input)
    
    if recipe_name:
        response = {
            "message": f"Here's a recipe for you!\n\nRecipe Name: {recipe_name}\n\nIngredients: {matched_ingredients}\n\nInstructions: {instructions}\n\nCooking Time: {cooking_time} mins"
        }
    else:
        response = {"message": "No matching recipe found and unable to generate a new recipe."}
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
