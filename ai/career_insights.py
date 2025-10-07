"""Career Decision Tree Algorithm for CareerSaathi AI."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from typing import Dict, List, Tuple, Optional
import joblib
import os

from services.utils import get_logger

logger = get_logger(__name__)


class CareerDecisionTree:
    """Decision tree model for career path recommendations."""
    
    def __init__(self, csv_path: str = "exp.csv"):
        self.csv_path = csv_path
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_encoder = LabelEncoder()
        self.career_data = None
        
    def _extract_salary_range(self, salary_str: str) -> Tuple[float, float]:
        """Extract min and max salary from salary range string."""
        if pd.isna(salary_str) or salary_str == "":
            return 0.0, 0.0
            
        # Remove commas and extract numbers
        numbers = re.findall(r'[\d,]+', str(salary_str))
        if not numbers:
            return 0.0, 0.0
            
        # Convert to float, removing commas
        nums = [float(num.replace(',', '')) for num in numbers]
        
        if len(nums) == 1:
            return nums[0], nums[0]
        elif len(nums) >= 2:
            return min(nums), max(nums)
        else:
            return 0.0, 0.0
    
    def _extract_growth_score(self, growth_str: str) -> float:
        """Convert growth outlook to numerical score."""
        if pd.isna(growth_str):
            return 0.0
            
        growth_str = str(growth_str).lower()
        
        if 'explosive' in growth_str:
            return 5.0
        elif 'exceptionally high' in growth_str or 'extremely high' in growth_str:
            return 4.5
        elif 'very high' in growth_str:
            return 4.0
        elif 'high' in growth_str:
            return 3.0
        elif 'faster than average' in growth_str:
            return 2.0
        elif 'average' in growth_str:
            return 1.0
        else:
            return 0.0
    
    def _categorize_role(self, role: str) -> str:
        """Categorize roles into major career tracks."""
        if pd.isna(role):
            return "Other"
            
        role_lower = str(role).lower()
        
        if any(keyword in role_lower for keyword in ['ai', 'ml', 'machine learning', 'data scientist', 'nlp']):
            return "AI_ML"
        elif any(keyword in role_lower for keyword in ['cloud', 'devops', 'sre', 'platform']):
            return "Cloud_Infrastructure"
        elif any(keyword in role_lower for keyword in ['data engineer', 'data analyst', 'analytics']):
            return "Data_Engineering"
        elif any(keyword in role_lower for keyword in ['software', 'developer', 'engineer', 'backend', 'frontend']):
            return "Software_Engineering"
        elif any(keyword in role_lower for keyword in ['product manager', 'product']):
            return "Product_Management"
        elif any(keyword in role_lower for keyword in ['security', 'cybersecurity']):
            return "Cybersecurity"
        elif any(keyword in role_lower for keyword in ['ux', 'ui', 'design', 'mobile']):
            return "Design_Mobile"
        else:
            return "Other"
    
    def preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the career data."""
        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_path, sep='|')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Skip header rows and empty rows
            df = df[df['Role'].notna() & (df['Role'] != 'Role') & (~df['Role'].str.contains(r'\*\*.*\*\*', na=False))]
            df = df[df['Role'].str.strip() != '']
            
            # Extract features
            processed_data = []
            
            for _, row in df.iterrows():
                role = row['Role']
                growth = row['Projected Growth Outlook (CAGR 2025-2030)']
                salary = row['Typical US Salary Range (USD)']
                skills = row['Required Core Skills to Master']
                
                # Extract salary range
                min_sal, max_sal = self._extract_salary_range(salary)
                avg_salary = (min_sal + max_sal) / 2 if max_sal > 0 else min_sal
                
                # Extract growth score
                growth_score = self._extract_growth_score(growth)
                
                # Categorize role
                role_category = self._categorize_role(role)
                
                # Count skills (rough proxy for complexity)
                skill_count = len(str(skills).split(',')) if pd.notna(skills) else 0
                
                # Determine experience level from role title
                exp_level = 0
                role_lower = str(role).lower()
                if any(word in role_lower for word in ['senior', 'principal', 'lead', 'director']):
                    exp_level = 2  # Senior
                elif any(word in role_lower for word in ['junior', 'entry', 'associate']):
                    exp_level = 0  # Entry
                else:
                    exp_level = 1  # Mid
                
                processed_data.append({
                    'role': role,
                    'role_category': role_category,
                    'growth_score': growth_score,
                    'avg_salary': avg_salary,
                    'skill_count': skill_count,
                    'experience_level': exp_level,
                    'salary_tier': 'High' if avg_salary > 150000 else 'Medium' if avg_salary > 100000 else 'Low'
                })
            
            processed_df = pd.DataFrame(processed_data)
            processed_df = processed_df.dropna()
            
            logger.info(f"Processed {len(processed_df)} career records")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train_model(self) -> None:
        """Train the decision tree model."""
        try:
            # Load and preprocess data
            self.career_data = self.preprocess_data()
            
            if len(self.career_data) == 0:
                raise ValueError("No valid data found for training")
            
            # Prepare features and target
            feature_columns = ['growth_score', 'avg_salary', 'skill_count', 'experience_level']
            X = self.career_data[feature_columns].copy()
            y = self.career_data['role_category'].copy()
            
            # Store feature names
            self.feature_names = feature_columns
            
            # Encode categorical target
            y_encoded = self.target_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train decision tree
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            logger.info(f"Feature importance: {dict(zip(self.feature_names, self.model.feature_importances_))}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict_career_path(self, user_profile: Dict) -> Dict:
        """Predict career path for a user profile."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        try:
            # Extract features from user profile
            features = {
                'growth_score': user_profile.get('growth_preference', 3.0),  # Default to high growth
                'avg_salary': user_profile.get('salary_expectation', 120000),  # Default to 120k
                'skill_count': len(user_profile.get('skills', [])),
                'experience_level': user_profile.get('experience_years', 0) // 3  # Convert years to level
            }
            
            # Create feature vector
            X = np.array([[features[col] for col in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get predicted category
            predicted_category = self.target_encoder.inverse_transform([prediction])[0]
            
            # Get top recommendations from the category
            category_roles = self.career_data[self.career_data['role_category'] == predicted_category]
            top_roles = category_roles.nlargest(3, 'avg_salary')[['role', 'avg_salary', 'growth_score']].to_dict('records')
            
            # Get confidence scores for all categories
            all_categories = self.target_encoder.classes_
            category_scores = {cat: prob for cat, prob in zip(all_categories, probabilities)}
            
            return {
                'primary_recommendation': predicted_category,
                'confidence': float(max(probabilities)),
                'top_roles': top_roles,
                'category_scores': category_scores,
                'reasoning': self._generate_reasoning(features, predicted_category)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _generate_reasoning(self, features: Dict, predicted_category: str) -> str:
        """Generate human-readable reasoning for the prediction."""
        reasoning_parts = []
        
        if features['experience_level'] == 0:
            reasoning_parts.append("As an entry-level candidate")
        elif features['experience_level'] == 1:
            reasoning_parts.append("With your mid-level experience")
        else:
            reasoning_parts.append("With your senior-level experience")
        
        if features['skill_count'] > 5:
            reasoning_parts.append("and diverse skill set")
        elif features['skill_count'] > 3:
            reasoning_parts.append("and solid skill foundation")
        else:
            reasoning_parts.append("and focused skill set")
        
        if features['avg_salary'] > 150000:
            reasoning_parts.append(f"targeting high compensation ({features['avg_salary']:,})")
        elif features['avg_salary'] > 100000:
            reasoning_parts.append(f"seeking competitive compensation ({features['avg_salary']:,})")
        
        category_descriptions = {
            'AI_ML': 'AI/ML roles offer exceptional growth potential and high compensation',
            'Cloud_Infrastructure': 'Cloud infrastructure roles are in extremely high demand',
            'Software_Engineering': 'Software engineering provides stable, well-compensated career paths',
            'Data_Engineering': 'Data engineering offers excellent growth in the data-driven economy',
            'Product_Management': 'Product management combines technical and business skills for leadership roles',
            'Cybersecurity': 'Cybersecurity roles are critical with exceptional job security',
            'Design_Mobile': 'Design and mobile development offer creative and technical opportunities'
        }
        
        reasoning = f"{', '.join(reasoning_parts)}, {category_descriptions.get(predicted_category, 'this career path')} is recommended."
        
        return reasoning
    
    def get_career_insights(self, role_category: str = None) -> Dict:
        """Get insights about career categories."""
        if self.career_data is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if role_category:
            category_data = self.career_data[self.career_data['role_category'] == role_category]
        else:
            category_data = self.career_data
        
        insights = {
            'avg_salary': float(category_data['avg_salary'].mean()),
            'salary_range': {
                'min': float(category_data['avg_salary'].min()),
                'max': float(category_data['avg_salary'].max())
            },
            'avg_growth_score': float(category_data['growth_score'].mean()),
            'role_count': len(category_data),
            'top_paying_roles': category_data.nlargest(3, 'avg_salary')[['role', 'avg_salary']].to_dict('records')
        }
        
        return insights
    
    def save_model(self, filepath: str = "career_model.joblib") -> None:
        """Save the trained model and encoders."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "career_model.joblib") -> None:
        """Load a pre-trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.target_encoder = model_data['target_encoder']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")


def create_user_profile_from_query(query: str) -> Dict:
    """Extract user profile information from a natural language query."""
    profile = {
        'experience_years': 0,
        'skills': [],
        'salary_expectation': 100000,
        'growth_preference': 3.0
    }
    
    query_lower = query.lower()
    
    # Extract experience
    import re
    exp_match = re.search(r'(\d+)\s*(?:years?|yrs?)', query_lower)
    if exp_match:
        profile['experience_years'] = int(exp_match.group(1))
    elif any(word in query_lower for word in ['entry', 'junior', 'new', 'fresh', 'graduate']):
        profile['experience_years'] = 0
    elif any(word in query_lower for word in ['senior', 'experienced', 'lead']):
        profile['experience_years'] = 8
    elif any(word in query_lower for word in ['mid', 'intermediate']):
        profile['experience_years'] = 4
    
    # Extract skills (basic keyword matching)
    skill_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker', 'kubernetes', 
                     'machine learning', 'data science', 'ai', 'cloud', 'devops', 'frontend', 'backend']
    
    for skill in skill_keywords:
        if skill in query_lower:
            profile['skills'].append(skill)
    
    # Extract salary expectations
    salary_match = re.search(r'\$?(\d+)k?', query_lower)
    if salary_match:
        salary = int(salary_match.group(1))
        if salary < 1000:  # Likely in thousands
            salary *= 1000
        profile['salary_expectation'] = salary
    
    # Extract growth preferences
    if any(word in query_lower for word in ['growth', 'fast-growing', 'emerging', 'cutting-edge']):
        profile['growth_preference'] = 4.5
    elif any(word in query_lower for word in ['stable', 'established', 'traditional']):
        profile['growth_preference'] = 2.0
    
    return profile