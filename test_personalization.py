#!/usr/bin/env python3
"""Test personalization of career recommendations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.career_recommendations import CareerRecommendationEngine

def test_personalization():
    """Test that different user profiles generate different recommendations."""
    print("🧪 Testing Career Recommendation Personalization")
    print("=" * 60)
    
    # Initialize engine
    engine = CareerRecommendationEngine()
    if not engine.initialize():
        print("❌ Failed to initialize recommendation engine")
        return
    
    # Test different user profiles
    test_profiles = [
        {
            'name': 'AI/ML Enthusiast',
            'profile': {
                'experience_years': 2,
                'skills': ['Python', 'Machine Learning', 'Data Science', 'TensorFlow'],
                'salary_expectation': 140000,
                'growth_preference': 4.5
            }
        },
        {
            'name': 'Cloud Engineer',
            'profile': {
                'experience_years': 5,
                'skills': ['AWS', 'Docker', 'Kubernetes', 'CI/CD'],
                'salary_expectation': 130000,
                'growth_preference': 4.0
            }
        },
        {
            'name': 'Frontend Developer',
            'profile': {
                'experience_years': 3,
                'skills': ['JavaScript', 'React', 'HTML/CSS', 'Node.js'],
                'salary_expectation': 110000,
                'growth_preference': 3.0
            }
        },
        {
            'name': 'Entry Level',
            'profile': {
                'experience_years': 0,
                'skills': ['Python', 'Git'],
                'salary_expectation': 80000,
                'growth_preference': 4.0
            }
        },
        {
            'name': 'Senior Executive',
            'profile': {
                'experience_years': 12,
                'skills': ['Project Management', 'Leadership', 'Strategy'],
                'salary_expectation': 200000,
                'growth_preference': 3.5
            }
        }
    ]
    
    for test_case in test_profiles:
        print(f"\n🎯 Testing: {test_case['name']}")
        profile = test_case['profile']
        
        print(f"   Skills: {', '.join(profile['skills'])}")
        print(f"   Experience: {profile['experience_years']} years")
        print(f"   Salary Target: ${profile['salary_expectation']:,}")
        print(f"   Growth Preference: {profile['growth_preference']}/5.0")
        
        # Get recommendations
        recommendations = engine.get_recommendations_from_profile(profile, num_recommendations=2)
        
        if recommendations:
            print(f"   ✅ Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec.role} ({rec.category.replace('_', ' ')})")
                print(f"         Salary: ${rec.avg_salary:,.0f} | Confidence: {rec.confidence:.0%}")
                print(f"         Skills to develop: {', '.join(rec.required_skills[:3])}")
        else:
            print(f"   ❌ No recommendations generated")
        
        print("-" * 60)
    
    print("\n🎉 Personalization test completed!")

if __name__ == "__main__":
    test_personalization()