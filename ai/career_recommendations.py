"""Career recommendation utilities for CareerSaathi."""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass

from services.utils import get_logger
from .career_insights import CareerDecisionTree, create_user_profile_from_query

logger = get_logger(__name__)


@dataclass
class CareerRecommendation:
    """Single career recommendation with details."""
    role: str
    category: str
    avg_salary: float
    growth_score: float
    confidence: float
    reasoning: str
    required_skills: List[str]
    next_steps: List[str]


class CareerRecommendationEngine:
    """High-level interface for career recommendations."""
    
    def __init__(self):
        self.decision_tree = CareerDecisionTree()
        self.is_trained = False
        
    def initialize(self) -> bool:
        """Initialize and train the recommendation engine."""
        try:
            import os
            model_path = "career_model.joblib"
            
            if os.path.exists(model_path):
                self.decision_tree.load_model(model_path)
                logger.info("Loaded existing career model")
            else:
                self.decision_tree.train_model()
                self.decision_tree.save_model(model_path)
                logger.info("Trained and saved new career model")
                
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            return False
    
    def get_recommendations(
        self, 
        user_input: str, 
        num_recommendations: int = 3
    ) -> List[CareerRecommendation]:
        """Get career recommendations based on user input."""
        if not self.is_trained:
            if not self.initialize():
                return []
        
        try:
            # Extract user profile
            user_profile = create_user_profile_from_query(user_input)
            
            # Get primary prediction
            prediction = self.decision_tree.predict_career_path(user_profile)
            
            # Get recommendations from multiple categories
            recommendations = []
            
            # Primary recommendation
            primary_rec = self._create_recommendation(
                prediction, 
                user_profile, 
                is_primary=True
            )
            if primary_rec:
                recommendations.append(primary_rec)
            
            # Alternative recommendations from other high-scoring categories
            sorted_categories = sorted(
                prediction['category_scores'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for category, score in sorted_categories[1:num_recommendations]:
                if score > 0.1:  # Only include if reasonably likely
                    alt_prediction = {
                        'primary_recommendation': category,
                        'confidence': score,
                        'top_roles': self._get_top_roles_for_category(category),
                        'reasoning': f"Alternative path with {score:.1%} likelihood"
                    }
                    alt_rec = self._create_recommendation(
                        alt_prediction, 
                        user_profile, 
                        is_primary=False
                    )
                    if alt_rec:
                        recommendations.append(alt_rec)
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def _create_recommendation(
        self, 
        prediction: Dict, 
        user_profile: Dict, 
        is_primary: bool = True
    ) -> Optional[CareerRecommendation]:
        """Create a CareerRecommendation from prediction data."""
        try:
            category = prediction['primary_recommendation']
            top_roles = prediction.get('top_roles', [])
            
            if not top_roles:
                return None
            
            # Use the highest-paying role as the main recommendation
            main_role = top_roles[0]
            
            # Get insights for the category
            insights = self.decision_tree.get_career_insights(category)
            
            # Generate required skills based on category
            required_skills = self._get_required_skills(category, user_profile)
            
            # Generate next steps
            next_steps = self._generate_next_steps(category, user_profile, is_primary)
            
            return CareerRecommendation(
                role=main_role['role'],
                category=category,
                avg_salary=main_role['avg_salary'],
                growth_score=main_role.get('growth_score', 3.0),
                confidence=prediction['confidence'],
                reasoning=prediction.get('reasoning', ''),
                required_skills=required_skills,
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.warning(f"Failed to create recommendation: {e}")
            return None
    
    def _get_top_roles_for_category(self, category: str) -> List[Dict]:
        """Get top roles for a specific category."""
        try:
            if self.decision_tree.career_data is not None:
                category_data = self.decision_tree.career_data[
                    self.decision_tree.career_data['role_category'] == category
                ]
                return category_data.nlargest(3, 'avg_salary')[
                    ['role', 'avg_salary', 'growth_score']
                ].to_dict('records')
        except Exception:
            pass
        return []
    
    def _get_required_skills(self, category: str, user_profile: Dict) -> List[str]:
        """Get required skills for a career category."""
        skill_map = {
            'AI_ML': ['Python', 'Machine Learning', 'Deep Learning', 'Statistics', 'TensorFlow/PyTorch'],
            'Cloud_Infrastructure': ['AWS/Azure/GCP', 'Kubernetes', 'Docker', 'Terraform', 'CI/CD'],
            'Software_Engineering': ['Programming Languages', 'System Design', 'Database Design', 'Testing', 'Git'],
            'Data_Engineering': ['SQL', 'Python/Scala', 'Big Data Tools', 'ETL/ELT', 'Data Modeling'],
            'Product_Management': ['Product Strategy', 'Market Analysis', 'Agile/Scrum', 'Stakeholder Management', 'Data Analysis'],
            'Cybersecurity': ['Security Frameworks', 'Network Security', 'Incident Response', 'Compliance', 'Scripting'],
            'Design_Mobile': ['UI/UX Design', 'Mobile Development', 'Design Tools', 'User Research', 'Prototyping']
        }
        
        base_skills = skill_map.get(category, ['Domain Knowledge', 'Problem Solving', 'Communication'])
        user_skills = set(user_profile.get('skills', []))
        
        # Add skills user doesn't have
        missing_skills = [skill for skill in base_skills if skill.lower() not in {s.lower() for s in user_skills}]
        
        return missing_skills[:5]  # Return top 5 missing skills
    
    def _generate_next_steps(self, category: str, user_profile: Dict, is_primary: bool) -> List[str]:
        """Generate actionable next steps for career development."""
        experience_years = user_profile.get('experience_years', 0)
        
        steps = []
        
        if experience_years == 0:
            steps.extend([
                "Build a portfolio showcasing relevant projects",
                "Complete online courses or certifications in core technologies",
                "Apply for entry-level positions or internships"
            ])
        elif experience_years < 3:
            steps.extend([
                "Gain hands-on experience with advanced tools and frameworks",
                "Seek mentorship from senior professionals in the field",
                "Consider lateral moves to gain broader experience"
            ])
        else:
            steps.extend([
                "Develop leadership and project management skills",
                "Build your professional network and personal brand",
                "Consider specializing in high-demand niches"
            ])
        
        # Category-specific steps
        category_steps = {
            'AI_ML': [
                "Complete machine learning projects with real datasets",
                "Contribute to open-source ML projects",
                "Stay updated with latest AI research and papers"
            ],
            'Cloud_Infrastructure': [
                "Obtain cloud certifications (AWS, Azure, or GCP)",
                "Practice with Infrastructure as Code tools",
                "Learn container orchestration and DevOps practices"
            ],
            'Software_Engineering': [
                "Master system design principles",
                "Contribute to open-source projects",
                "Practice coding interviews and algorithm problems"
            ]
        }
        
        if category in category_steps:
            steps.extend(category_steps[category][:2])
        
        if not is_primary:
            steps.insert(0, "Research this alternative career path thoroughly")
        
        return steps[:4]  # Return top 4 steps
    
    def get_salary_insights(self, role_category: str = None) -> Dict:
        """Get salary insights for career categories."""
        if not self.is_trained:
            return {}
        
        return self.decision_tree.get_career_insights(role_category)
    
    def analyze_career_gaps(self, user_input: str) -> Dict:
        """Analyze gaps between current profile and target career."""
        if not self.is_trained:
            return {}
        
        try:
            user_profile = create_user_profile_from_query(user_input)
            prediction = self.decision_tree.predict_career_path(user_profile)
            
            target_category = prediction['primary_recommendation']
            required_skills = self._get_required_skills(target_category, user_profile)
            current_skills = user_profile.get('skills', [])
            
            skill_gaps = [skill for skill in required_skills if skill.lower() not in {s.lower() for s in current_skills}]
            
            insights = self.decision_tree.get_career_insights(target_category)
            salary_gap = max(0, insights['avg_salary'] - user_profile.get('salary_expectation', 0))
            
            return {
                'target_category': target_category,
                'skill_gaps': skill_gaps,
                'estimated_salary_increase': salary_gap,
                'development_timeline': self._estimate_timeline(len(skill_gaps), user_profile['experience_years']),
                'priority_skills': skill_gaps[:3]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze career gaps: {e}")
            return {}
    
    def _estimate_timeline(self, skill_gaps: int, experience_years: int) -> str:
        """Estimate timeline for career transition."""
        base_months = 6  # Base time for any transition
        
        # Add time based on skill gaps
        skill_months = skill_gaps * 2
        
        # Adjust based on experience
        if experience_years == 0:
            multiplier = 1.5  # Takes longer for beginners
        elif experience_years > 5:
            multiplier = 0.8  # Experienced professionals learn faster
        else:
            multiplier = 1.0
        
        total_months = int((base_months + skill_months) * multiplier)
        
        if total_months <= 6:
            return "3-6 months"
        elif total_months <= 12:
            return "6-12 months"
        elif total_months <= 18:
            return "12-18 months"
        else:
            return "18+ months"


# Global instance for easy access
recommendation_engine = CareerRecommendationEngine()


def get_quick_career_advice(user_query: str) -> str:
    """Get quick career advice in text format."""
    recommendations = recommendation_engine.get_recommendations(user_query, num_recommendations=1)
    
    if not recommendations:
        return "I'm sorry, I couldn't generate specific career recommendations. Please provide more details about your background and career goals."
    
    rec = recommendations[0]
    
    advice = f"""Based on your profile, I recommend pursuing a role as a **{rec.role}** in the {rec.category.replace('_', ' ')} field.

**Why this recommendation:**
{rec.reasoning}

**Key details:**
- Average salary: ${rec.avg_salary:,.0f}
- Growth potential: {'High' if rec.growth_score > 3.5 else 'Moderate' if rec.growth_score > 2.5 else 'Stable'}
- Match confidence: {rec.confidence:.0%}

**Skills to develop:**
{', '.join(rec.required_skills[:3])}

**Next steps:**
1. {rec.next_steps[0] if rec.next_steps else 'Start building relevant skills'}
2. {rec.next_steps[1] if len(rec.next_steps) > 1 else 'Network with professionals in the field'}
3. {rec.next_steps[2] if len(rec.next_steps) > 2 else 'Apply for relevant positions'}"""

    return advice