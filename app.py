from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import spacy

app = Flask(__name__)
CORS(app)  # This will allow CORS for all routes

@app.route('/', methods=['POST'])
def student_project_mapping():
    nlp = spacy.load('en_core_web_md')

    data = request.json
    student_info = "This student has studied modules in {}. They have previous experience in {} and have a preference for projects related to {}."

    # Fill in the placeholders using string formatting
    filled_student_info = student_info.format(data['modules_taken'], data['previous_experience'],data['project_preference'])

    projects = data['data']

    # Preprocess student information
    student_doc = nlp(filled_student_info.lower())

    # Preprocess project information and create embeddings
    project_docs = []
    project_embeddings = []
    for project in projects:
        project_doc = nlp(project["Description"].lower() + " " + " ".join(project["Skill"]).lower())
        project_docs.append(project_doc)
        project_embeddings.append(project_doc.vector)

    # Calculate similarity between student info and projects
    similarities = [cosine_similarity(student_doc.vector.reshape(1, -1), project_embedding.reshape(1, -1))[0][0] for
                    project_embedding in project_embeddings]

    # Rank projects based on similarity
    ranked_projects = sorted(zip(projects, similarities), key=lambda x: x[1], reverse=True)

    # Recommend top 5 projects
    top_projects = ranked_projects[:5]

    # Extract titles of top recommended projects
    recommended_titles = [project['Title'] for project, _ in top_projects]
    print(recommended_titles)

    # Find matching projects in the main project array and add additional fields
    recommended_projects_with_additional_fields = []
    for recommended_title in recommended_titles:
        # Find the project with the matching title
        matching_project = next((project for project in projects if project['Title'] == recommended_title), None)
        if matching_project:
            # Add additional fields to the recommended project
            recommended_project_with_additional_fields = {
                'Title': matching_project['Title'],
                'Description': matching_project['Description'],
                'Skill': matching_project['Skill'],
                'Difficulty_Rating': matching_project['Difficulty_Rating'],  # Additional field
                'Module_ID': matching_project['Module_ID'],  # Additional field
                'Program_ID': matching_project['Program_ID'],  # Additional field
                'Programe_List': matching_project['Programe_List'],  # Additional field
                'S_email': matching_project['S_email'],  # Additional field
                'S_name': matching_project['S_name'],  # Additional field
                'Staff_ID': matching_project['Staff_ID'],  # Additional field
                'Project_ID': matching_project['Project_ID']  # Additional field
            }
            recommended_projects_with_additional_fields.append(recommended_project_with_additional_fields)
    return jsonify(recommended_projects_with_additional_fields)

if __name__ == '__main__':
    app.run(debug=True)