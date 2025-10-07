"""Career Insights page using the decision tree algorithm."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List

from ai.career_recommendations import CareerRecommendationEngine, get_quick_career_advice
from ai.career_insights import create_user_profile_from_query
from streamlit_app.components import session as session_components
from services.utils import get_logger

logger = get_logger(__name__)

# Initialize session and require authentication
session_components.initialise_session()
user = session_components.require_authentication()

st.title("🎯 AI-Powered Career Insights")
st.markdown("*Discover your ideal career path using data-driven recommendations*")

# Initialize recommendation engine
@st.cache_resource
def get_recommendation_engine():
    """Get cached recommendation engine instance."""
    engine = CareerRecommendationEngine()
    if engine.initialize():
        return engine
    return None

engine = get_recommendation_engine()

if not engine:
    st.error("❌ Career insights are currently unavailable. Please try again later.")
    st.stop()

# Sidebar for user input
st.sidebar.header("📝 Tell us about yourself")

with st.sidebar:
    # Experience level
    experience_years = st.selectbox(
        "Years of Experience",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
        index=2,
        help="Select your years of professional experience"
    )
    
    # Skills
    st.subheader("🛠️ Skills")
    skill_categories = {
        "Programming": ["Python", "Java", "JavaScript", "C++", "Go", "Rust", "C#"],
        "Data & AI": ["Machine Learning", "Data Science", "SQL", "Deep Learning", "Statistics"],
        "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD"],
        "Web Development": ["React", "Angular", "Node.js", "HTML/CSS", "REST APIs"],
        "Other": ["Project Management", "Design", "Mobile Development", "Cybersecurity"]
    }
    
    selected_skills = []
    for category, skills in skill_categories.items():
        with st.expander(f"{category} Skills"):
            for skill in skills:
                if st.checkbox(skill, key=f"skill_{skill}"):
                    selected_skills.append(skill)
    
    # Salary expectation
    salary_expectation = st.slider(
        "💰 Salary Expectation (USD)",
        min_value=50000,
        max_value=300000,
        value=100000,
        step=10000,
        help="Your target annual salary"
    )
    
    # Growth preference
    growth_preference = st.select_slider(
        "📈 Growth Preference",
        options=["Stable", "Moderate", "High", "Very High", "Explosive"],
        value="High",
        help="How important is career growth potential to you?"
    )
    
    # Industry interests
    industries = [
        "Technology", "Finance", "Healthcare", "Education", "Consulting",
        "Government", "Startups", "Enterprise", "Research", "Gaming"
    ]
    preferred_industries = st.multiselect(
        "🏢 Industry Interests",
        industries,
        default=["Technology"],
        help="Industries you're interested in working in"
    )

# Create user profile
growth_map = {"Stable": 1.0, "Moderate": 2.0, "High": 3.0, "Very High": 4.0, "Explosive": 5.0}
user_profile = {
    'experience_years': experience_years,
    'skills': selected_skills,
    'salary_expectation': salary_expectation,
    'growth_preference': growth_map[growth_preference],
    'industries': preferred_industries
}

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "📊 Market Insights", "📈 Career Gap Analysis", "🤖 AI Chat"])

# Show a quick preview of the current profile
with st.expander("📋 Your Profile Summary", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Experience", f"{experience_years} years")
        st.metric("Skills Count", len(selected_skills))
    with col2:
        st.metric("Salary Target", f"${salary_expectation:,}")
        st.metric("Growth Focus", growth_preference)
    with col3:
        st.write("**Top Skills:**")
        for skill in selected_skills[:3]:
            st.markdown(f"• {skill}")
        if len(selected_skills) > 3:
            st.markdown(f"• +{len(selected_skills) - 3} more")

with tab1:
    st.header("Your Personalized Career Recommendations")
    
    # Add auto-generate option
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔮 Generate Recommendations", type="primary"):
            generate_recs = True
        else:
            generate_recs = False
            
    with col2:
        auto_generate = st.checkbox("Auto-update", help="Automatically update recommendations when you change your profile")
    
    # Generate recommendations if button clicked or auto-update is enabled
    if generate_recs or (auto_generate and selected_skills):
        with st.spinner("Analyzing your profile and generating recommendations..."):
            try:
                # Use the user profile directly instead of converting to text
                recommendations = engine.get_recommendations_from_profile(user_profile, num_recommendations=3)
                
                if recommendations:
                    st.success(f"✅ Generated {len(recommendations)} personalized recommendations based on your profile!")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"### {i}. {rec.role}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("💰 Average Salary", f"${rec.avg_salary:,.0f}")
                            with col2:
                                st.metric("📈 Growth Score", f"{rec.growth_score:.1f}/5.0")
                            with col3:
                                st.metric("🎯 Confidence", f"{rec.confidence:.0%}")
                            
                            st.markdown(f"**Category:** {rec.category.replace('_', ' ')}")
                            st.markdown(f"**Why this fits:** {rec.reasoning}")
                            
                            # Show user's current skills vs required
                            if selected_skills:
                                st.markdown("**✅ Your current skills:**")
                                skill_cols = st.columns(min(len(selected_skills), 4))
                                for j, skill in enumerate(selected_skills[:4]):
                                    with skill_cols[j % 4]:
                                        st.markdown(f"• {skill}")
                            
                            # Skills needed
                            if rec.required_skills:
                                st.markdown("**🛠️ Skills to develop:**")
                                skill_cols = st.columns(min(len(rec.required_skills), 3))
                                for j, skill in enumerate(rec.required_skills[:3]):
                                    with skill_cols[j % 3]:
                                        st.markdown(f"• {skill}")
                            
                            # Next steps
                            if rec.next_steps:
                                st.markdown("**📋 Next steps:**")
                                for step in rec.next_steps[:3]:
                                    st.markdown(f"1. {step}")
                            
                            st.divider()
                else:
                    st.warning("No recommendations could be generated. Please adjust your profile and try again.")
                    
            except Exception as e:
                logger.error(f"Failed to generate recommendations: {e}")
                st.error("Failed to generate recommendations. Please try again.")
    
    elif not selected_skills:
        st.info("👈 Please select some skills in the sidebar to get personalized recommendations!")

with tab2:
    st.header("📊 Career Market Insights")
    
    # Show salary insights by category
    try:
        categories = ['AI_ML', 'Cloud_Infrastructure', 'Software_Engineering', 'Data_Engineering', 
                     'Product_Management', 'Cybersecurity', 'Design_Mobile']
        
        insights_data = []
        for category in categories:
            insights = engine.get_salary_insights(category)
            if insights:
                insights_data.append({
                    'Category': category.replace('_', ' '),
                    'Average Salary': insights['avg_salary'],
                    'Min Salary': insights['salary_range']['min'],
                    'Max Salary': insights['salary_range']['max'],
                    'Role Count': insights['role_count']
                })
        
        if insights_data:
            df = pd.DataFrame(insights_data)
            
            # Salary comparison chart
            fig = px.bar(
                df, 
                x='Category', 
                y='Average Salary',
                title='Average Salary by Career Category',
                color='Average Salary',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Salary range chart
            fig2 = go.Figure()
            
            for _, row in df.iterrows():
                fig2.add_trace(go.Scatter(
                    x=[row['Min Salary'], row['Max Salary']],
                    y=[row['Category'], row['Category']],
                    mode='lines+markers',
                    name=row['Category'],
                    line=dict(width=8),
                    marker=dict(size=10)
                ))
            
            fig2.update_layout(
                title='Salary Ranges by Career Category',
                xaxis_title='Salary (USD)',
                yaxis_title='Career Category',
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Data table
            st.subheader("📋 Detailed Insights")
            st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        logger.error(f"Failed to generate market insights: {e}")
        st.error("Failed to load market insights.")

with tab3:
    st.header("📈 Career Gap Analysis")
    
    if st.button("🔍 Analyze My Career Gaps", type="primary"):
        with st.spinner("Analyzing your career development needs..."):
            try:
                query = f"I have {experience_years} years experience with skills in {', '.join(selected_skills)} seeking ${salary_expectation} salary"
                gaps = engine.analyze_career_gaps(query)
                
                if gaps:
                    st.markdown(f"### Target Career: {gaps['target_category'].replace('_', ' ')}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "💰 Potential Salary Increase", 
                            f"${gaps['estimated_salary_increase']:,.0f}"
                        )
                        st.metric(
                            "⏱️ Estimated Timeline", 
                            gaps['development_timeline']
                        )
                    
                    with col2:
                        if gaps['skill_gaps']:
                            st.markdown("**🎯 Priority Skills to Develop:**")
                            for skill in gaps['priority_skills']:
                                st.markdown(f"• {skill}")
                        else:
                            st.success("🎉 You already have most required skills!")
                    
                    if gaps['skill_gaps']:
                        st.markdown("**📚 All Skill Gaps:**")
                        skill_cols = st.columns(3)
                        for i, skill in enumerate(gaps['skill_gaps']):
                            with skill_cols[i % 3]:
                                st.markdown(f"• {skill}")
                else:
                    st.warning("Could not analyze career gaps. Please ensure your profile is complete.")
                    
            except Exception as e:
                logger.error(f"Failed to analyze career gaps: {e}")
                st.error("Failed to analyze career gaps.")

with tab4:
    st.header("🤖 AI Career Chat")
    st.markdown("Ask me anything about your career path!")
    
    # Chat input
    user_question = st.text_area(
        "Your Question:",
        placeholder="E.g., 'What's the best way to transition from data analysis to machine learning?' or 'Should I pursue a cloud certification?'",
        height=100
    )
    
    if st.button("💬 Get AI Advice", type="primary") and user_question:
        with st.spinner("Generating personalized career advice..."):
            try:
                # Enhance the question with user profile context
                enhanced_query = f"{user_question}. Context: I have {experience_years} years experience with skills in {', '.join(selected_skills[:5])} and salary expectation of ${salary_expectation}."
                
                advice = get_quick_career_advice(enhanced_query)
                
                if advice:
                    st.markdown("### 💡 AI Career Advice")
                    st.markdown(advice)
                else:
                    st.warning("I couldn't generate advice for your question. Please try rephrasing it.")
                    
            except Exception as e:
                logger.error(f"Failed to generate AI advice: {e}")
                st.error("Failed to generate career advice. Please try again.")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Tips for Better Recommendations:
- **Be specific about your skills** - Include both technical and soft skills
- **Update your experience level** - Accurate experience helps with better matching
- **Consider multiple recommendations** - Sometimes alternative paths offer great opportunities
- **Focus on skill gaps** - Developing missing skills can significantly boost your career prospects
""")

st.markdown("""
### 📊 Data Sources:
This career recommendation system analyzes current market data including:
- Salary ranges from industry reports
- Growth projections from labor statistics
- Skill requirements from job postings
- Career progression patterns

*Last updated: 2025*
""")