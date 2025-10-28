"""Quick test to verify skill matching is working correctly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.career_insights import CareerDecisionTree

def test_skill_matching():
    """Test that JavaScript/HTML skills give Software Engineering roles"""
    
    print("="*70)
    print("🧪 TESTING SKILL MATCHING - JavaScript/HTML/CSS")
    print("="*70)
    
    # Initialize and train
    tree = CareerDecisionTree()
    print("\n📚 Training model...")
    tree.train_model()
    print("✅ Model trained successfully!")
    
    # Test Case 1: Web Developer Skills
    print("\n" + "="*70)
    print("TEST 1: Web Developer Skills (JS, HTML, CSS, React)")
    print("="*70)
    
    web_dev_profile = {
        'experience_years': 3,
        'skills': ['JavaScript', 'HTML/CSS', 'React', 'Node.js'],
        'salary_expectation': 100000,
        'growth_preference': 3.5
    }
    
    result = tree.predict_career_path(web_dev_profile)
    
    if result:
        print(f"\n✅ Primary Recommendation: {result['primary_recommendation']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n🎯 Top 3 Roles:")
        for i, role in enumerate(result['top_roles'][:3], 1):
            print(f"   {i}. {role['role']}")
            print(f"      Salary: ${role['avg_salary']:,.0f}")
            print(f"      Growth: {role['growth_score']:.1f}/5.0")
        
        print(f"\n📊 All Category Scores:")
        sorted_scores = sorted(result['category_scores'].items(), key=lambda x: x[1], reverse=True)
        for cat, score in sorted_scores[:5]:
            print(f"   {cat:25} {score:.2%}")
        
        # Check if Software Engineering is top
        if result['primary_recommendation'] == 'Software_Engineering':
            print("\n✅ SUCCESS: Web dev skills correctly matched to Software Engineering!")
        else:
            print(f"\n❌ FAIL: Expected Software_Engineering but got {result['primary_recommendation']}")
    else:
        print("❌ No prediction returned")
    
    # Test Case 2: UI/UX Designer Skills
    print("\n" + "="*70)
    print("TEST 2: UI/UX Designer Skills")
    print("="*70)
    
    design_profile = {
        'experience_years': 2,
        'skills': ['UI', 'UX', 'Design', 'Mobile Development'],
        'salary_expectation': 90000,
        'growth_preference': 3.0
    }
    
    result = tree.predict_career_path(design_profile)
    
    if result:
        print(f"\n✅ Primary Recommendation: {result['primary_recommendation']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n🎯 Top Role: {result['top_roles'][0]['role']}")
        
        if result['primary_recommendation'] == 'Design_Mobile':
            print("\n✅ SUCCESS: Design skills correctly matched to Design_Mobile!")
        else:
            print(f"\n⚠️  Got {result['primary_recommendation']} instead of Design_Mobile")
    
    # Test Case 3: AI/ML Skills
    print("\n" + "="*70)
    print("TEST 3: AI/ML Skills (Python, ML, TensorFlow)")
    print("="*70)
    
    ml_profile = {
        'experience_years': 4,
        'skills': ['Python', 'Machine Learning', 'TensorFlow', 'Deep Learning'],
        'salary_expectation': 130000,
        'growth_preference': 4.5
    }
    
    result = tree.predict_career_path(ml_profile)
    
    if result:
        print(f"\n✅ Primary Recommendation: {result['primary_recommendation']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n🎯 Top Role: {result['top_roles'][0]['role']}")
        
        if result['primary_recommendation'] == 'AI_ML':
            print("\n✅ SUCCESS: ML skills correctly matched to AI_ML!")
        else:
            print(f"\n⚠️  Got {result['primary_recommendation']} instead of AI_ML")
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)

if __name__ == "__main__":
    test_skill_matching()
