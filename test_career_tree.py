"""Test script for career decision tree functionality."""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.career_insights import CareerDecisionTree, create_user_profile_from_query
from ai.career_recommendations import CareerRecommendationEngine, get_quick_career_advice


def test_decision_tree():
    """Test the basic decision tree functionality."""
    print("🧪 Testing Career Decision Tree...")
    
    try:
        # Initialize and train model
        dt = CareerDecisionTree()
        dt.train_model()
        
        # Test prediction
        test_profile = {
            'experience_years': 2,
            'skills': ['python', 'machine learning'],
            'salary_expectation': 120000,
            'growth_preference': 4.0
        }
        
        prediction = dt.predict_career_path(test_profile)
        print(f"✅ Prediction successful: {prediction['primary_recommendation']}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Reasoning: {prediction['reasoning']}")
        
        # Test insights
        insights = dt.get_career_insights('AI_ML')
        print(f"✅ Insights for AI/ML: Avg salary ${insights['avg_salary']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision tree test failed: {e}")
        return False


def test_user_profile_extraction():
    """Test user profile extraction from natural language."""
    print("\n🧪 Testing User Profile Extraction...")
    
    test_queries = [
        "I'm a software engineer with 3 years experience in Python and React, looking for $130k salary",
        "Fresh graduate with machine learning skills seeking high-growth career",
        "Senior developer with 8 years experience interested in cloud technologies"
    ]
    
    for query in test_queries:
        try:
            profile = create_user_profile_from_query(query)
            print(f"✅ Query: {query[:50]}...")
            print(f"   Profile: {profile}")
            
        except Exception as e:
            print(f"❌ Profile extraction failed for query: {e}")
            return False
    
    return True


def test_recommendation_engine():
    """Test the high-level recommendation engine."""
    print("\n🧪 Testing Recommendation Engine...")
    
    try:
        engine = CareerRecommendationEngine()
        success = engine.initialize()
        
        if not success:
            print("❌ Failed to initialize recommendation engine")
            return False
        
        # Test recommendations
        query = "I have 2 years experience in Python and want to work in AI"
        recommendations = engine.get_recommendations(query, num_recommendations=2)
        
        if recommendations:
            print(f"✅ Generated {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec.role} (${rec.avg_salary:,.0f}) - {rec.confidence:.1%}")
        else:
            print("❌ No recommendations generated")
            return False
        
        # Test gap analysis
        gaps = engine.analyze_career_gaps(query)
        if gaps:
            print(f"✅ Gap analysis: {len(gaps.get('skill_gaps', []))} skill gaps identified")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation engine test failed: {e}")
        return False


def test_quick_advice():
    """Test the quick advice function."""
    print("\n🧪 Testing Quick Career Advice...")
    
    try:
        query = "I'm a data analyst with SQL and Python skills, want to transition to machine learning"
        advice = get_quick_career_advice(query)
        
        if advice and len(advice) > 100:
            print("✅ Quick advice generated successfully")
            print(f"   Length: {len(advice)} characters")
            print(f"   Preview: {advice[:200]}...")
        else:
            print("❌ Quick advice generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quick advice test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Career Decision Tree Test Suite")
    print("=" * 50)
    
    tests = [
        test_decision_tree,
        test_user_profile_extraction,
        test_recommendation_engine,
        test_quick_advice
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! The career decision tree is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return all(results)


if __name__ == "__main__":
    main()