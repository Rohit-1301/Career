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
        if pd.isna(salary_str) or salary_str == "" or str(salary_str).strip() == "":
            return 0.0, 0.0
            
        # Clean the string and extract numbers
        salary_clean = str(salary_str).replace(',', '').replace('$', '').replace('USD', '')
        
        # Look for number patterns (including decimals)
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', salary_clean)
        
        if not numbers:
            return 0.0, 0.0
            
        # Convert to float
        try:
            nums = [float(num) for num in numbers]
        except ValueError:
            return 0.0, 0.0
        
        # Handle different formats
        if len(nums) == 1:
            # Single number - check if it's in thousands (e.g., "150" means 150k)
            val = nums[0]
            if val < 1000:  # Likely in thousands
                val *= 1000
            return val, val
        elif len(nums) >= 2:
            # Range - take first two numbers
            min_val, max_val = nums[0], nums[1]
            # Convert to proper scale if needed
            if min_val < 1000:
                min_val *= 1000
            if max_val < 1000:
                max_val *= 1000
            return min_val, max_val
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
        
        # AI/ML roles
        if any(keyword in role_lower for keyword in ['ai', 'ml', 'machine learning', 'data scientist', 'nlp', 'computer vision', 'research scientist']):
            return "AI_ML"
        # Cloud and DevOps
        elif any(keyword in role_lower for keyword in ['cloud', 'devops', 'sre', 'platform', 'infrastructure', 'site reliability']):
            return "Cloud_Infrastructure"
        # Data Engineering and Analytics
        elif any(keyword in role_lower for keyword in ['data engineer', 'data analyst', 'analytics', 'etl', 'big data', 'data warehouse']):
            return "Data_Engineering"
        # Software Engineering (broader definition)
        elif any(keyword in role_lower for keyword in ['software', 'developer', 'engineer', 'backend', 'frontend', 'full stack', 'programming']):
            return "Software_Engineering"
        # Product Management
        elif any(keyword in role_lower for keyword in ['product manager', 'product']):
            return "Product_Management"
        # Cybersecurity
        elif any(keyword in role_lower for keyword in ['security', 'cybersecurity']):
            return "Cybersecurity"
        # Design and Mobile
        elif any(keyword in role_lower for keyword in ['ux', 'ui', 'design', 'mobile']):
            return "Design_Mobile"
        # Leadership and Management
        elif any(keyword in role_lower for keyword in ['director', 'manager', 'lead', 'principal', 'head of', 'vp', 'cto', 'ceo']):
            return "Leadership"
        # Sales and Business
        elif any(keyword in role_lower for keyword in ['sales', 'business', 'marketing', 'account', 'consultant']):
            return "Business"
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
            
            # Remove rows that are just separators or empty
            df = df[~df['Role'].str.contains(r'^[-|]+$', na=False)]
            df = df[df['Role'].str.len() > 3]  # Remove very short entries
            
            # Extract features
            processed_data = []
            
            for _, row in df.iterrows():
                try:
                    role = str(row['Role']).strip()
                    growth = str(row.get('Projected Growth Outlook (CAGR 2025-2030)', '')) if 'Projected Growth Outlook (CAGR 2025-2030)' in row else ''
                    salary = str(row.get('Typical US Salary Range (USD)', '')) if 'Typical US Salary Range (USD)' in row else ''
                    skills = str(row.get('Required Core Skills to Master', '')) if 'Required Core Skills to Master' in row else ''
                    
                    # Skip if essential data is missing
                    if not role or role in ['Role', ''] or 'AI/ML' in role:
                        continue
                    
                    # Extract salary range
                    min_sal, max_sal = self._extract_salary_range(salary)
                    if min_sal == 0 and max_sal == 0:
                        # Assign default salaries based on role keywords
                        role_lower = role.lower()
                        if any(word in role_lower for word in ['senior', 'principal', 'director', 'lead']):
                            min_sal, max_sal = 150000, 250000
                        elif any(word in role_lower for word in ['engineer', 'developer', 'scientist']):
                            min_sal, max_sal = 100000, 180000
                        else:
                            min_sal, max_sal = 80000, 120000
                    
                    avg_salary = (min_sal + max_sal) / 2 if max_sal > 0 else min_sal
                    
                    # Extract growth score
                    growth_score = self._extract_growth_score(growth)
                    if growth_score == 0:
                        growth_score = 3.0  # Default to moderate growth
                    
                    # Categorize role
                    role_category = self._categorize_role(role)
                    
                    # Count skills (rough proxy for complexity)
                    skill_count = len(str(skills).split(',')) if pd.notna(skills) and skills else 3
                    
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
                    
                except Exception as e:
                    logger.warning(f"Skipping row due to error: {e}")
                    continue
            
            processed_df = pd.DataFrame(processed_data)
            processed_df = processed_df.dropna()
            
            # Ensure we have enough data
            if len(processed_df) < 10:
                logger.warning("Insufficient data found, creating synthetic data")
                processed_df = self._create_synthetic_data()
            
            logger.info(f"Processed {len(processed_df)} career records")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            # Return synthetic data as fallback
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic career data as fallback."""
        synthetic_data = [
            {'role': 'Software Engineer', 'role_category': 'Software_Engineering', 'growth_score': 3.5, 'avg_salary': 120000, 'skill_count': 5, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Senior Software Engineer', 'role_category': 'Software_Engineering', 'growth_score': 3.5, 'avg_salary': 160000, 'skill_count': 6, 'experience_level': 2, 'salary_tier': 'High'},
            {'role': 'Data Scientist', 'role_category': 'AI_ML', 'growth_score': 4.5, 'avg_salary': 140000, 'skill_count': 6, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Machine Learning Engineer', 'role_category': 'AI_ML', 'growth_score': 4.5, 'avg_salary': 150000, 'skill_count': 7, 'experience_level': 1, 'salary_tier': 'High'},
            {'role': 'Cloud Engineer', 'role_category': 'Cloud_Infrastructure', 'growth_score': 4.0, 'avg_salary': 130000, 'skill_count': 5, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'DevOps Engineer', 'role_category': 'Cloud_Infrastructure', 'growth_score': 4.0, 'avg_salary': 125000, 'skill_count': 6, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Product Manager', 'role_category': 'Product_Management', 'growth_score': 3.5, 'avg_salary': 135000, 'skill_count': 4, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'UX Designer', 'role_category': 'Design_Mobile', 'growth_score': 3.0, 'avg_salary': 110000, 'skill_count': 4, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Cybersecurity Analyst', 'role_category': 'Cybersecurity', 'growth_score': 4.0, 'avg_salary': 115000, 'skill_count': 5, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Data Engineer', 'role_category': 'Data_Engineering', 'growth_score': 4.0, 'avg_salary': 125000, 'skill_count': 5, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Frontend Developer', 'role_category': 'Software_Engineering', 'growth_score': 3.0, 'avg_salary': 105000, 'skill_count': 4, 'experience_level': 0, 'salary_tier': 'Medium'},
            {'role': 'Backend Developer', 'role_category': 'Software_Engineering', 'growth_score': 3.5, 'avg_salary': 115000, 'skill_count': 5, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'Full Stack Developer', 'role_category': 'Software_Engineering', 'growth_score': 3.5, 'avg_salary': 118000, 'skill_count': 6, 'experience_level': 1, 'salary_tier': 'Medium'},
            {'role': 'AI Research Scientist', 'role_category': 'AI_ML', 'growth_score': 5.0, 'avg_salary': 200000, 'skill_count': 8, 'experience_level': 2, 'salary_tier': 'High'},
            {'role': 'Senior Data Engineer', 'role_category': 'Data_Engineering', 'growth_score': 4.0, 'avg_salary': 165000, 'skill_count': 6, 'experience_level': 2, 'salary_tier': 'High'},
        ]
        
        return pd.DataFrame(synthetic_data)
    
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
            
            # Remove categories with too few samples
            category_counts = y.value_counts()
            min_samples_per_category = 3  # Minimum samples needed
            valid_categories = category_counts[category_counts >= min_samples_per_category].index
            
            if len(valid_categories) < 2:
                logger.warning("Not enough categories with sufficient samples, using synthetic data")
                self.career_data = self._create_synthetic_data()
                X = self.career_data[feature_columns].copy()
                y = self.career_data['role_category'].copy()
            else:
                # Filter data to only include valid categories
                mask = y.isin(valid_categories)
                X = X[mask]
                y = y[mask]
                self.career_data = self.career_data[mask]
            
            # Store feature names
            self.feature_names = feature_columns
            
            # Encode categorical target
            y_encoded = self.target_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Check if we have enough data points for train-test split
            unique_classes = len(np.unique(y_encoded))
            if len(X_scaled) < unique_classes * 2:
                # Not enough data for stratified split, train on all data
                X_train, X_test, y_train, y_test = X_scaled, X_scaled, y_encoded, y_encoded
                logger.warning("Training on all data due to small dataset size")
            else:
                # Split data with stratification
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
                'growth_score': user_profile.get('growth_preference', 3.0),
                'avg_salary': user_profile.get('salary_expectation', 120000),
                'skill_count': len(user_profile.get('skills', [])),
                'experience_level': min(user_profile.get('experience_years', 0) // 3, 2)  # 0-2 scale
            }
            
            # Create feature vector
            X = np.array([[features[col] for col in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get predicted category
            predicted_category = self.target_encoder.inverse_transform([prediction])[0]
            
            # Adjust predictions based on user skills
            adjusted_scores = self._adjust_predictions_by_skills(
                dict(zip(self.target_encoder.classes_, probabilities)),
                user_profile.get('skills', [])
            )
            
            # Re-sort categories by adjusted scores
            top_category = max(adjusted_scores.items(), key=lambda x: x[1])
            predicted_category = top_category[0]
            
            # Get top recommendations from the category
            if self.career_data is not None:
                category_roles = self.career_data[self.career_data['role_category'] == predicted_category]
                if len(category_roles) > 0:
                    top_roles = category_roles.nlargest(3, 'avg_salary')[['role', 'avg_salary', 'growth_score']].to_dict('records')
                else:
                    top_roles = self._get_fallback_roles(predicted_category)
            else:
                # Fallback if career_data is not available
                top_roles = self._get_fallback_roles(predicted_category)
            
            return {
                'primary_recommendation': predicted_category,
                'confidence': float(adjusted_scores[predicted_category]),
                'top_roles': top_roles,
                'category_scores': adjusted_scores,
                'reasoning': self._generate_reasoning(features, predicted_category, user_profile)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _adjust_predictions_by_skills(self, category_scores: Dict[str, float], user_skills: List[str]) -> Dict[str, float]:
        """Adjust category predictions based on user's skills."""
        if not user_skills:
            return category_scores
        
        # Define skill-to-category mappings
        skill_category_map = {
            'python': ['AI_ML', 'Data_Engineering', 'Software_Engineering'],
            'java': ['Software_Engineering'],
            'javascript': ['Software_Engineering'],
            'react': ['Software_Engineering'],
            'angular': ['Software_Engineering'],
            'node.js': ['Software_Engineering'],
            'machine learning': ['AI_ML'],
            'data science': ['AI_ML', 'Data_Engineering'],
            'sql': ['Data_Engineering', 'AI_ML'],
            'aws': ['Cloud_Infrastructure'],
            'azure': ['Cloud_Infrastructure'],
            'gcp': ['Cloud_Infrastructure'],
            'docker': ['Cloud_Infrastructure', 'Software_Engineering'],
            'kubernetes': ['Cloud_Infrastructure'],
            'devops': ['Cloud_Infrastructure'],
            'ci/cd': ['Cloud_Infrastructure'],
            'design': ['Design_Mobile'],
            'ui': ['Design_Mobile'],
            'ux': ['Design_Mobile'],
            'mobile development': ['Design_Mobile'],
            'ios': ['Design_Mobile'],
            'android': ['Design_Mobile'],
            'cybersecurity': ['Cybersecurity'],
            'project management': ['Product_Management'],
            'product management': ['Product_Management']
        }
        
        adjusted_scores = category_scores.copy()
        
        # Boost categories that match user skills
        for skill in user_skills:
            skill_lower = skill.lower()
            for skill_key, categories in skill_category_map.items():
                if skill_key in skill_lower or skill_lower in skill_key:
                    for category in categories:
                        if category in adjusted_scores:
                            # Boost the score by 20-50% based on skill relevance
                            boost = 0.3 if skill_key == skill_lower else 0.2
                            adjusted_scores[category] = min(1.0, adjusted_scores[category] + boost)
        
        # Normalize scores
        total_score = sum(adjusted_scores.values())
        if total_score > 0:
            for category in adjusted_scores:
                adjusted_scores[category] /= total_score
        
        return adjusted_scores
    
    def _get_fallback_roles(self, category: str) -> List[Dict]:
        """Get fallback role recommendations when career_data is not available."""
        fallback_roles = {
            'AI_ML': [
                {'role': 'Data Scientist', 'avg_salary': 140000, 'growth_score': 4.5},
                {'role': 'Machine Learning Engineer', 'avg_salary': 150000, 'growth_score': 4.5},
                {'role': 'AI Research Scientist', 'avg_salary': 200000, 'growth_score': 5.0}
            ],
            'Software_Engineering': [
                {'role': 'Software Engineer', 'avg_salary': 120000, 'growth_score': 3.5},
                {'role': 'Senior Software Engineer', 'avg_salary': 160000, 'growth_score': 3.5},
                {'role': 'Staff Software Engineer', 'avg_salary': 220000, 'growth_score': 3.5}
            ],
            'Cloud_Infrastructure': [
                {'role': 'Cloud Engineer', 'avg_salary': 130000, 'growth_score': 4.0},
                {'role': 'DevOps Engineer', 'avg_salary': 125000, 'growth_score': 4.0},
                {'role': 'Platform Engineer', 'avg_salary': 145000, 'growth_score': 4.0}
            ],
            'Data_Engineering': [
                {'role': 'Data Engineer', 'avg_salary': 125000, 'growth_score': 4.0},
                {'role': 'Senior Data Engineer', 'avg_salary': 165000, 'growth_score': 4.0},
                {'role': 'Staff Data Engineer', 'avg_salary': 200000, 'growth_score': 4.0}
            ],
            'Product_Management': [
                {'role': 'Product Manager', 'avg_salary': 135000, 'growth_score': 3.5},
                {'role': 'Senior Product Manager', 'avg_salary': 175000, 'growth_score': 3.5},
                {'role': 'Principal Product Manager', 'avg_salary': 220000, 'growth_score': 3.5}
            ],
            'Cybersecurity': [
                {'role': 'Cybersecurity Analyst', 'avg_salary': 115000, 'growth_score': 4.0},
                {'role': 'Security Engineer', 'avg_salary': 140000, 'growth_score': 4.0},
                {'role': 'Security Architect', 'avg_salary': 180000, 'growth_score': 4.0}
            ]
        }
        
        return fallback_roles.get(category, [
            {'role': 'Professional', 'avg_salary': 100000, 'growth_score': 3.0},
            {'role': 'Senior Professional', 'avg_salary': 140000, 'growth_score': 3.0},
            {'role': 'Expert Professional', 'avg_salary': 180000, 'growth_score': 3.0}
        ])
    
    def _generate_reasoning(self, features: Dict, predicted_category: str, user_profile: Dict = None) -> str:
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
        
        # Add specific skills if available
        if user_profile and user_profile.get('skills'):
            top_skills = user_profile['skills'][:3]
            reasoning_parts.append(f"particularly in {', '.join(top_skills)}")
        
        if features['avg_salary'] > 150000:
            reasoning_parts.append(f"targeting high compensation (${features['avg_salary']:,})")
        elif features['avg_salary'] > 100000:
            reasoning_parts.append(f"seeking competitive compensation (${features['avg_salary']:,})")
        
        category_descriptions = {
            'AI_ML': 'AI/ML roles offer exceptional growth potential and high compensation',
            'Cloud_Infrastructure': 'Cloud infrastructure roles are in extremely high demand',
            'Software_Engineering': 'Software engineering provides stable, well-compensated career paths',
            'Data_Engineering': 'Data engineering offers excellent growth in the data-driven economy',
            'Product_Management': 'Product management combines technical and business skills for leadership roles',
            'Cybersecurity': 'Cybersecurity roles are critical with exceptional job security',
            'Design_Mobile': 'Design and mobile development offer creative and technical opportunities',
            'Leadership': 'Leadership roles provide opportunities to guide teams and strategy',
            'Business': 'Business roles offer diverse opportunities across industries'
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
        
        # Reload career data for insights
        try:
            self.career_data = self.preprocess_data()
        except Exception as e:
            logger.warning(f"Could not reload career data: {e}")
            self.career_data = None
        
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