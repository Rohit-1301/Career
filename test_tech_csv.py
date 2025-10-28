"""Test script to verify career insights work with tech.csv"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.career_insights import CareerDecisionTree

def test_tech_csv_loading():
    """Test that tech.csv loads correctly"""
    print("=" * 60)
    print("Testing Career Insights with tech.csv")
    print("=" * 60)
    
    # Initialize the decision tree with tech.csv (default)
    tree = CareerDecisionTree()
    
    print("\n1. Loading and preprocessing data from tech.csv...")
    try:
        df = tree.preprocess_data()
        print(f"✓ Successfully loaded {len(df)} career records")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df.head())
        print(f"\nRole categories found: {df['role_category'].unique()}")
        print(f"Category distribution:\n{df['role_category'].value_counts()}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Training the model...")
    try:
        tree.train_model()
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing predictions...")
    # Test AI/ML enthusiast
    test_profiles = [
        {
            'name': 'AI/ML Enthusiast',
            'profile': {
                'experience_years': 2,
                'skills': ['python', 'machine learning', 'tensorflow', 'data science'],
                'salary_expectation': 120000,
                'growth_preference': 4.5
            }
        },
        {
            'name': 'Cloud Engineer',
            'profile': {
                'experience_years': 4,
                'skills': ['aws', 'docker', 'kubernetes', 'terraform'],
                'salary_expectation': 130000,
                'growth_preference': 4.0
            }
        },
        {
            'name': 'Data Engineer',
            'profile': {
                'experience_years': 3,
                'skills': ['python', 'sql', 'spark', 'airflow'],
                'salary_expectation': 115000,
                'growth_preference': 4.0
            }
        }
    ]
    
    for test in test_profiles:
        print(f"\n--- Testing: {test['name']} ---")
        try:
            result = tree.predict_career_path(test['profile'])
            if result:
                print(f"✓ Primary Recommendation: {result['primary_recommendation']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Top Roles:")
                for role in result['top_roles'][:3]:
                    print(f"    - {role['role']}: ${role['avg_salary']:,.0f} (Growth: {role['growth_score']:.1f}/5)")
                print(f"  Reasoning: {result['reasoning']}")
            else:
                print(f"✗ No prediction returned")
        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n4. Testing career insights...")
    try:
        insights = tree.get_career_insights()
        print(f"✓ Average salary across all roles: ${insights['avg_salary']:,.0f}")
        print(f"  Salary range: ${insights['salary_range']['min']:,.0f} - ${insights['salary_range']['max']:,.0f}")
        print(f"  Average growth score: {insights['avg_growth_score']:.2f}/5")
        print(f"  Total roles: {insights['role_count']}")
    except Exception as e:
        print(f"✗ Error getting insights: {e}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_tech_csv_loading()
    sys.exit(0 if success else 1)
