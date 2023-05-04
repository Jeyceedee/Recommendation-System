from flask import Flask, request, jsonify
import os
import boto3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel

# Set up AWS credentials and DynamoDB client
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAZRHCIEMCGV5NERU4"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-1"

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Campaign-bv5ga2wk7fecbijb73l3zh3t4e-staging')

# Function to fetch all items from the DynamoDB table
def scan_table(table):
    response = table.scan()
    items = response['Items']
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])
    return items

# Fetch data from DynamoDB and convert it to a pandas DataFrame
items = scan_table(table)
data = pd.DataFrame(items)

# Load pre-trained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get DistilBERT embeddings for a given text
def get_distilbert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Create DistilBERT embeddings for campaign purposes
purpose_embeddings = np.vstack(data['description'].apply(lambda x: get_distilbert_embeddings(x, tokenizer, model)))

# Calculate cosine similarity between campaign purposes
cosine_sim = cosine_similarity(purpose_embeddings, purpose_embeddings)

# Define a function to get similar campaigns based on campaign purpose
# Define a function to get similar campaigns based on campaign purpose
def get_similar_campaigns(title, cosine_sim=cosine_sim, data=data):
    title_matches = data[data['campaignName'] == title]
    if len(title_matches) > 0:
        idx = title_matches.index[0]
    else:
        print('Campaign "{}" not found in data'.format(title))
        return pd.Series(dtype=str)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    campaign_indices = [i[0] for i in sim_scores]
    result = data[['id', 'campaignName']].iloc[campaign_indices]
    return result.to_dict(orient='records')


app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({"error": "title parameter is required"}), 400

    try:
        similar_campaigns = get_similar_campaigns(title)
        return jsonify({"recommended_campaigns": similar_campaigns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
