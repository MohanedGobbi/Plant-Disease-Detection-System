import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

class SymptomMatcher:
    def __init__(self, prompts_file):
        self.prompts_file = prompts_file
        self.prompts_data = self._load_prompts()
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sbert_embeddings = None
        self.disease_info = []
        
        try:
            self._prepare_data()
            self._fit_tfidf()
            self._fit_sbert()
            print("SymptomMatcher initialized successfully!")
        except Exception as e:
            print(f"Error initializing SymptomMatcher: {e}")
            # Create minimal fallback data
            self.disease_info = [{
                'plant': 'unknown',
                'disease': 'unknown',
                'symptoms': 'no symptoms available',
                'description': 'no description available',
                'text_for_matching': 'unknown plant disease'
            }]
            self._fit_tfidf()
            self._fit_sbert()
    
    def _load_prompts(self):
        """Load plant disease prompts from JSON file"""
        with open(self.prompts_file, 'r') as f:
            return json.load(f)
    
    def _preprocess_text(self, text):
        """Preprocess text for similarity matching - ensure it returns a string"""
        if isinstance(text, list):
            text = " ".join([str(item) for item in text])  # Convert all items to string
        text = str(text).lower()  # Ensure it's a string
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text
    
    def _extract_plant_name(self, disease_name):
        """Extract plant name from disease name"""
        plant_names = ['apple', 'banana', 'basil', 'bean', 'bell pepper', 'blueberry', 
                      'broccoli', 'cabbage', 'carrot', 'cauliflower', 'celery', 'cherry',
                      'citrus', 'coffee', 'corn', 'cucumber', 'eggplant', 'garlic', 
                      'ginger', 'grape', 'tomato', 'potato', 'strawberry', 'pepper']
        
        disease_lower = str(disease_name).lower()
        for plant in plant_names:
            if disease_lower.startswith(plant):
                return plant
        return disease_lower.split()[0] if disease_lower.split() else "unknown"
    
    def _prepare_data(self):
        """Prepare and normalize data from JSON - create disease_info list"""
        self.disease_info = []
        
        print(f"Processing {len(self.prompts_data)} diseases from JSON...")
        
        for disease_name, descriptions in self.prompts_data.items():
            try:
                # Handle different data structures more robustly
                if isinstance(descriptions, list):
                    # Flatten nested lists if they exist
                    flat_descriptions = []
                    for desc in descriptions:
                        if isinstance(desc, list):
                            flat_descriptions.extend([str(item) for item in desc])
                        else:
                            flat_descriptions.append(str(desc))
                    combined_descriptions = " ".join(flat_descriptions)
                elif isinstance(descriptions, dict):
                    # Handle dictionary format
                    combined_descriptions = " ".join([str(v) for v in descriptions.values()])
                else:
                    # Handle string or other formats
                    combined_descriptions = str(descriptions)
                
                # Clean up the text
                combined_descriptions = combined_descriptions.strip()
                if not combined_descriptions:
                    combined_descriptions = "no description available"
                
                plant_name = self._extract_plant_name(disease_name)
                text_for_matching = f"{plant_name} {disease_name} {combined_descriptions}"
                
                self.disease_info.append({
                    'plant': plant_name,
                    'disease': disease_name,
                    'symptoms': combined_descriptions,
                    'description': combined_descriptions,
                    'text_for_matching': text_for_matching
                })
                
            except Exception as e:
                print(f"Warning: Error processing disease '{disease_name}': {e}")
                # Add a fallback entry
                plant_name = self._extract_plant_name(disease_name)
                self.disease_info.append({
                    'plant': plant_name,
                    'disease': disease_name,
                    'symptoms': 'symptoms unavailable',
                    'description': 'description unavailable',
                    'text_for_matching': f"{plant_name} {disease_name} symptoms unavailable"
                })
        
        print(f"Prepared {len(self.disease_info)} disease entries")
    
    def _fit_tfidf(self):
        """Fit TF-IDF vectorizer on all symptom texts"""
        # Use the preprocessed text from disease_info
        texts = []
        
        for info in self.disease_info:
            text = info['text_for_matching']
            
            # Triple-check that we have a string
            if isinstance(text, list):
                text = " ".join([str(item) for item in text])
            elif not isinstance(text, str):
                text = str(text)
            
            # Clean the text to remove any problematic characters
            text = re.sub(r'[^\w\s]', ' ', text)  # Replace non-alphanumeric with space
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
            
            if not text:  # If text is empty after cleaning
                text = "no description available"
            
            texts.append(text)
        
        print(f"Fitting TF-IDF with {len(texts)} documents...")
        
        # Double-check all texts are strings and not empty
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                print(f"Warning: Converting non-string text at index {i}: {type(text)}")
                texts[i] = str(text)
            if not text.strip():
                print(f"Warning: Empty text at index {i}, replacing with placeholder")
                texts[i] = "no content available"
        
        try:
            # Fit and transform
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print(f"TF-IDF fitted successfully with {self.tfidf_matrix.shape[0]} documents and {self.tfidf_matrix.shape[1]} features")
        except Exception as e:
            print(f"Error fitting TF-IDF: {e}")
            print("Sample texts:")
            for i, text in enumerate(texts[:3]):
                print(f"  {i}: {type(text)} - {repr(text[:100])}")
            raise
    
    def _fit_sbert(self):
        """Precompute SBERT embeddings for all symptoms"""
        # Use the same preprocessed text as TF-IDF
        texts = [info['text_for_matching'] for info in self.disease_info]
        
        # Compute SBERT embeddings
        self.sbert_embeddings = self.sbert_model.encode(texts, convert_to_tensor=True)
        print(f"SBERT embeddings computed for {len(texts)} documents")
    
    def _tfidf_match(self, query, top_k=5):
        """Find matches using TF-IDF cosine similarity"""
        query = self._preprocess_text(query)
        query_vec = self.tfidf_vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0 and idx < len(self.disease_info):
                disease_info = self.disease_info[idx]
                results.append({
                    'plant': disease_info['plant'],
                    'disease': disease_info['disease'],
                    'symptoms': disease_info['symptoms'],
                    'description': disease_info['description'],
                    'similarity': float(similarities[idx]),
                    'method': 'TF-IDF'
                })
        
        return results
    
    def _sbert_match(self, query, top_k=5):
        """Find matches using SBERT cosine similarity"""
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.sbert_embeddings)[0]
        
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx].item()
            if similarity > 0 and idx < len(self.disease_info):
                disease_info = self.disease_info[idx]
                results.append({
                    'plant': disease_info['plant'],
                    'disease': disease_info['disease'],
                    'symptoms': disease_info['symptoms'],
                    'description': disease_info['description'],
                    'similarity': similarity,
                    'method': 'SBERT'
                })
        
        return results
    
    def match_symptoms(self, query, method='both', top_k=5):
        """Find diseases with symptoms similar to the query"""
        if not query or not isinstance(query, str) or not query.strip():
            return []
            
        try:
            if method == 'tfidf':
                return self._tfidf_match(query, top_k)
            elif method == 'sbert':
                return self._sbert_match(query, top_k)
            elif method == 'both':
                tfidf_results = self._tfidf_match(query, top_k)
                sbert_results = self._sbert_match(query, top_k)
                
                combined = {}
                for result in tfidf_results + sbert_results:
                    key = (result['plant'], result['disease'])
                    if key not in combined or result['similarity'] > combined[key]['similarity']:
                        combined[key] = result
                
                sorted_results = sorted(combined.values(), key=lambda x: x['similarity'], reverse=True)
                return sorted_results[:top_k]
            else:
                raise ValueError("Method must be 'tfidf', 'sbert', or 'both'")
        except Exception as e:
            print(f"Error in match_symptoms: {e}")
            return []

# Add debug function to check your JSON structure
def debug_json_structure(filename):
    """Debug function to check your JSON file structure"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print("JSON Structure Analysis:")
        print(f"Number of diseases: {len(data)}")
        
        for i, (disease_name, descriptions) in enumerate(data.items()):
            print(f"\n{i+1}. {disease_name}:")
            print(f"   Type of descriptions: {type(descriptions)}")
            if isinstance(descriptions, list):
                print(f"   Number of descriptions: {len(descriptions)}")
                for j, desc in enumerate(descriptions[:2]):  # Show first 2
                    print(f"     {j+1}. Type: {type(desc)}, Preview: {str(desc)[:50]}...")
            else:
                print(f"   Content: {str(descriptions)[:100]}...")
                
    except Exception as e:
        print(f"Error loading JSON: {e}")

# Example usage with debugging
if __name__ == "__main__":
    # First debug your JSON structure
    debug_json_structure("data\plantwild_prompts.json")
    
    # Then initialize the matcher
    matcher = SymptomMatcher("data\plantwild_prompts.json")
    
    # Test with a query
    query = "circular black spots on fruit with concentric rings"
    results = matcher.match_symptoms(query, method='both', top_k=5)
    
    print(f"\nQuery: '{query}'")
    print("-" * 80)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['plant']} - {result['disease']} (Similarity: {result['similarity']:.3f}, Method: {result['method']})")
        print(f"   Symptoms: {result['symptoms'][:150]}...")
        print()