from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SMSCategorization:
    def __init__(self):
        # Initialize the model and categories
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.categories = [
            "Expenses related to food and drink.",
            "Travel-related activities, tickets, and bookings.",
            "Utility bills like electricity, water, or gas.",
            "Credit transactions such as account deposits or withdrawals.",
            "Money transfer or transactions.",
            "Entertainment activities like movies, shows, or subscriptions.",
            "Shopping-related expenses or e-commerce orders."
        ]
        # Generate embeddings for the categories
        self.category_embeddings = self._generate_embeddings(self.categories)

    def _generate_embeddings(self, texts):
        """Generate normalized embeddings for a list of texts."""
        embeddings = np.array([self.model.encode(text) for text in texts])
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def categorize_sms(self, sms_messages):
        """Categorize a list of SMS messages."""
        sms_embeddings = self._generate_embeddings(sms_messages)
        results = []

        for sms, sms_embedding in zip(sms_messages, sms_embeddings):
            category = self._match_category(sms_embedding)
            results.append({"sms": sms, "category": category})

        return results

    def _match_category(self, sms_embedding):
        """Find the best matching category for a given SMS embedding."""
        similarities = cosine_similarity([sms_embedding], self.category_embeddings)
        best_match_idx = np.argmax(similarities)
        return self.categories[best_match_idx]

