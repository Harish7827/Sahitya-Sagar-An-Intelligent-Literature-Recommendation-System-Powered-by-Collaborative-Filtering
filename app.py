from flask import Flask, render_template, request
import pickle
import numpy as np
from rapidfuzz import process, fuzz

# Load models and data
pbr_df = pickle.load(open('Model/PopularBookRecommendation.pkl', 'rb'))
pt = pickle.load(open('Model/pt.pkl', 'rb'))
book = pickle.load(open('Model/book.pkl', 'rb'))
similarity_scores = pickle.load(open('Model/similarity_scores.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Render homepage with top popular books
    return render_template(
        'index.html',
        book_name=list(pbr_df['Book-Title'].values),
        author=list(pbr_df['Book-Author'].values),
        publisher=list(pbr_df['Publisher'].values),
        image=list(pbr_df['Image-URL-L'].values),
        votes=list(pbr_df['Num_rating'].values),
        rating=list(pbr_df['Avg_rating'].values),
    )

@app.route('/recommendation')
def recommendation_ui():
    return render_template('recommendation.html')

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    user_input = request.form.get('user_input')

    # Fuzzy match for book name
    matches = process.extract(user_input, pt.index, limit=5, scorer=fuzz.partial_ratio)
    if not matches:
        return render_template('recommendation.html', error="No similar book found. Please try again.")
    
    best_match, score, _ = matches[0]
    if score < 50:
        return render_template('recommendation.html', error="No close match found. Please try again.")

    # Get recommendations
    index = np.where(pt.index == best_match)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:9]

    data = []
    for i in similar_items:
        item = []
        temp_df = book[book['Book-Title'] == pt.index[i[0]]]
        if not temp_df.empty:
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-L'].values))
            data.append(item)

    return render_template('recommendation.html', data=data, suggested=best_match)

@app.route('/book/<book_name>')
def book_detail(book_name):
    try:
        temp_df = book[book['Book-Title'] == book_name].drop_duplicates('Book-Title')
        if temp_df.empty:
            return render_template('book_detail.html', error="Book details not available.")

        book_data = {
            'title': temp_df['Book-Title'].values[0],
            'author': temp_df['Book-Author'].values[0],
            'publisher': temp_df['Publisher'].values[0],
            'image': temp_df['Image-URL-L'].values[0]
        }

        # Fetch recommendations
        if book_name in pt.index:
            index = np.where(pt.index == book_name)[0][0]
            similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:9]

            recommendations = []
            for i in similar_items:
                item_df = book[book['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
                if not item_df.empty:
                    recommendations.append({
                        'title': item_df['Book-Title'].values[0],
                        'author': item_df['Book-Author'].values[0],
                        'image': item_df['Image-URL-L'].values[0]
                    })
        else:
            recommendations = []

        return render_template('book_detail.html', book=book_data, recommendations=recommendations)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('book_detail.html', error="Something went wrong. Please try again.")

# 404 Error Page
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
