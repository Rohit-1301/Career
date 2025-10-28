"""Career Insights page using the decision tree algorithm."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

from ai.career_recommendations import CareerRecommendationEngine, get_quick_career_advice
from ai.career_insights import create_user_profile_from_query
from streamlit_app.components import session as session_components
from services.utils import get_logger

logger = get_logger(__name__)

# Currency conversion helper
USD_TO_INR = 83  # 1 USD ≈ 83 INR

def format_inr_salary(usd_salary):
    """Convert USD salary to INR and format it."""
    inr_salary = usd_salary * USD_TO_INR
    if inr_salary >= 10000000:  # 1 Crore or more
        return f"₹{inr_salary/10000000:.2f} Cr"
    elif inr_salary >= 100000:  # 1 Lakh or more
        return f"₹{inr_salary/100000:.2f} LPA"
    else:
        return f"₹{inr_salary:,.0f}"

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
    
    # Salary expectation in INR
    salary_expectation_inr = st.slider(
        "💰 Salary Expectation (INR per year)",
        min_value=300000,      # 3 LPA
        max_value=10000000,    # 1 Crore
        value=1200000,         # 12 LPA
        step=100000,           # 1 Lakh steps
        help="Your target annual salary in Indian Rupees",
        format="₹%d"
    )
    
    # Convert INR to USD for internal calculations (1 USD ≈ 83 INR)
    salary_expectation = salary_expectation_inr / 83
    
    # Growth preference
    growth_preference = st.select_slider(
        "📈 Growth Preference",
        options=["Stable", "Moderate", "High", "Very High", "Explosive"],
        value="High",
        help="How important is career growth potential to you?"
    )

# Create user profile
growth_map = {"Stable": 1.0, "Moderate": 2.0, "High": 3.0, "Very High": 4.0, "Explosive": 5.0}
user_profile = {
    'experience_years': experience_years,
    'skills': selected_skills,
    'salary_expectation': salary_expectation,
    'growth_preference': growth_map[growth_preference]
}

# Main content area
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Market Insights", "📈 Career Gap Analysis"])

# Show a quick preview of the current profile
with st.expander("📋 Your Profile Summary", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Experience", f"{experience_years} years")
        st.metric("Skills Count", len(selected_skills))
    with col2:
        st.metric("Salary Target", f"₹{salary_expectation_inr:,.0f}")
        st.metric("Growth Focus", growth_preference)
    with col3:
        st.write("**Top Skills:**")
        for skill in selected_skills[:3]:
            st.markdown(f"• {skill}")
        if len(selected_skills) > 3:
            st.markdown(f"• +{len(selected_skills) - 3} more")

with tab1:
    st.header("Your Personalized Career Recommendations")
    
    # Check if user has provided sufficient information
    if not selected_skills:
        st.warning("⚠️ **Please add your skills to get personalized recommendations!**")
        st.info("""
        ### 📝 To get started:
        1. **Select your skills** from the sidebar (under "🛠️ Skills")
        2. **Set your experience level** to help us understand your background
        3. **Adjust your salary expectations** to match your goals
        4. **Choose your growth preference** to find the right career path
        
        Once you've completed your profile, recommendations will appear automatically!
        """)
        st.image("https://via.placeholder.com/600x300/4ECDC4/FFFFFF?text=Add+Your+Skills+to+Get+Started", use_container_width=True)
    else:
        # Validate profile completeness
        if len(selected_skills) < 2:
            st.warning("⚠️ **Please add at least 2 skills for better recommendations!**")
            st.info("""
            We need more information about your skills to provide accurate career recommendations.
            
            **Please add more skills from the sidebar to continue.**
            """)
        else:
            with st.spinner("🔮 Analyzing your profile and generating personalized recommendations..."):
                try:
                    # Use the user profile directly instead of converting to text
                    recommendations = engine.get_recommendations_from_profile(user_profile, num_recommendations=10)
                    
                    if recommendations:
                        st.success(f"✅ Generated {len(recommendations)} personalized recommendations based on your unique profile!")
                        
                        # Show dynamic insights based on selected skills
                        st.info(f"""
                        💡 **Smart Match**: Based on your expertise in **{', '.join(selected_skills[:3])}**{' and ' + str(len(selected_skills) - 3) + ' more skills' if len(selected_skills) > 3 else ''}, 
                        we've identified the most relevant career paths with **{format_inr_salary(recommendations[0].avg_salary)}** average salary potential.
                        """)
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.container():
                                # Make recommendation headers more prominent with colors
                                if i == 1:
                                    st.markdown(f"### 🥇 {i}. {rec.role}")
                                elif i == 2:
                                    st.markdown(f"### 🥈 {i}. {rec.role}")
                                else:
                                    st.markdown(f"### 🥉 {i}. {rec.role}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("💰 Average Salary", format_inr_salary(rec.avg_salary))
                                with col2:
                                    growth_emoji = "🚀" if rec.growth_score > 4 else "📈" if rec.growth_score > 3 else "➡️"
                                    st.metric(f"{growth_emoji} Growth Score", f"{rec.growth_score:.1f}/5.0")
                                with col3:
                                    confidence_color = "🟢" if rec.confidence > 0.6 else "🟡" if rec.confidence > 0.3 else "🔴"
                                    st.metric(f"{confidence_color} Match", f"{rec.confidence:.0%}")
                                with col4:
                                    category_emoji = {
                                        'AI_ML': '🤖',
                                        'Cloud_Infrastructure': '☁️',
                                        'Software_Engineering': '💻',
                                        'Data_Engineering': '📊',
                                        'Product_Management': '📦',
                                        'Cybersecurity': '🔒',
                                        'Design_Mobile': '📱'
                                    }
                                    emoji = category_emoji.get(rec.category, '🎯')
                                    st.metric(f"{emoji} Category", rec.category.replace('_', ' '))
                                
                                st.markdown(f"**💡 Why this fits:** {rec.reasoning}")
                                
                                # Create interactive skill comparison
                                col_skills_1, col_skills_2 = st.columns(2)
                                
                                with col_skills_1:
                                    st.markdown("**✅ Your Current Skills:**")
                                    if selected_skills:
                                        # Show skills in a nicer format with badges
                                        skill_tags = " ".join([f"`{skill}`" for skill in selected_skills[:6]])
                                        st.markdown(skill_tags)
                                        if len(selected_skills) > 6:
                                            st.markdown(f"*+{len(selected_skills) - 6} more skills*")
                                    
                                with col_skills_2:
                                    # Skills needed
                                    if rec.required_skills:
                                        st.markdown("**🛠️ Skills to Develop:**")
                                        skill_tags = " ".join([f"`{skill}`" for skill in rec.required_skills[:6]])
                                        st.markdown(skill_tags)
                                        if len(rec.required_skills) > 6:
                                            st.markdown(f"*+{len(rec.required_skills) - 6} more skills*")
                                
                                # Interactive expandable section for next steps
                                with st.expander(f"📋 See Your Action Plan for {rec.role}", expanded=(i==1)):
                                    if rec.next_steps:
                                        st.markdown("**🎯 Recommended Next Steps:**")
                                        for idx, step in enumerate(rec.next_steps, 1):
                                            st.markdown(f"{idx}. {step}")
                                    
                                    # Add interactive checkboxes for tracking progress
                                    st.markdown("---")
                                    st.markdown("**📝 Track Your Progress:**")
                                    for idx, step in enumerate(rec.next_steps[:3], 1):
                                        st.checkbox(f"Complete: {step[:50]}...", key=f"progress_{rec.role}_{idx}")
                                
                                st.divider()
                        
                        # Add comparison chart for recommendations
                        if len(recommendations) > 1:
                            st.subheader("📊 Compare Your Recommendations")
                            
                            comparison_data = {
                                'Role': [rec.role for rec in recommendations],
                                'Salary (INR)': [rec.avg_salary * USD_TO_INR for rec in recommendations],
                                'Salary (LPA)': [rec.avg_salary * USD_TO_INR / 100000 for rec in recommendations],
                                'Growth': [rec.growth_score for rec in recommendations],
                                'Match %': [rec.confidence * 100 for rec in recommendations]
                            }
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Create interactive comparison chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                name='Salary (in LPA)',
                                x=comparison_df['Role'],
                                y=comparison_df['Salary (LPA)'],
                                marker_color='#4ECDC4',
                                yaxis='y',
                                offsetgroup=1,
                                hovertemplate='<b>%{x}</b><br>Salary: ₹%{y:.1f} LPA<extra></extra>'
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='Growth Score (×3)',
                                x=comparison_df['Role'],
                                y=comparison_df['Growth'] * 3,
                                marker_color='#FF6B6B',
                                yaxis='y',
                                offsetgroup=2,
                                hovertemplate='<b>%{x}</b><br>Growth: %{customdata:.1f}/5.0<extra></extra>',
                                customdata=comparison_df['Growth']
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='Match % (×0.2)',
                                x=comparison_df['Role'],
                                y=comparison_df['Match %'] * 0.2,
                                marker_color='#45B7D1',
                                yaxis='y',
                                offsetgroup=3,
                                hovertemplate='<b>%{x}</b><br>Match: %{customdata:.0f}%<extra></extra>',
                                customdata=comparison_df['Match %']
                            ))
                            
                            fig.update_layout(
                                title='Quick Comparison: Salary (LPA) vs Growth vs Match',
                                xaxis_title='Career Role',
                                yaxis_title='Value (LPA / Score)',
                                barmode='group',
                                height=600,  # Increased height for more roles
                                showlegend=True,
                                hovermode='x unified',
                                xaxis_tickangle=-45  # Angle labels for better readability
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add summary statistics
                            st.markdown("### 📈 Quick Stats")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_salary = sum(rec.avg_salary for rec in recommendations) / len(recommendations)
                                st.metric("Average Salary (All Paths)", format_inr_salary(avg_salary))
                            with col2:
                                avg_growth = sum(rec.growth_score for rec in recommendations) / len(recommendations)
                                st.metric("Average Growth Score", f"{avg_growth:.1f}/5.0")
                            with col3:
                                high_match = sum(1 for rec in recommendations if rec.confidence > 0.5)
                                st.metric("High Match Roles", f"{high_match}/{len(recommendations)}")
                            
                    else:
                        st.warning("No recommendations could be generated. Please adjust your profile and try again.")
                        
                except Exception as e:
                    logger.error(f"Failed to generate recommendations: {e}")
                    import traceback
                    traceback.print_exc()
                    st.error("Failed to generate recommendations. Please try again or adjust your profile.")


with tab2:
    st.header("📊 Career Market Insights")
    
    # Get comprehensive market data
    try:
        # Market overview metrics
        st.subheader("🎯 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Categories", "7", help="Major career categories tracked")
        with col2:
            st.metric("Avg Salary Range", "₹6.6 - ₹16.6 LPA", help="Typical salary range across all categories in Indian Rupees")
        with col3:
            st.metric("High Growth Areas", "3", help="Categories with >4.0 growth score")
        with col4:
            st.metric("Entry Points", "15+", help="Roles suitable for entry-level candidates")
        
        # Comprehensive market data
        categories_data = {
            'AI_ML': {'avg_salary': 155000, 'min_salary': 120000, 'max_salary': 220000, 'growth': 4.8, 'demand': 'Very High', 'roles': 12},
            'Cloud_Infrastructure': {'avg_salary': 135000, 'min_salary': 100000, 'max_salary': 180000, 'growth': 4.5, 'demand': 'High', 'roles': 15},
            'Software_Engineering': {'avg_salary': 125000, 'min_salary': 85000, 'max_salary': 200000, 'growth': 3.8, 'demand': 'High', 'roles': 25},
            'Data_Engineering': {'avg_salary': 140000, 'min_salary': 110000, 'max_salary': 190000, 'growth': 4.2, 'demand': 'High', 'roles': 10},
            'Product_Management': {'avg_salary': 145000, 'min_salary': 105000, 'max_salary': 220000, 'growth': 3.5, 'demand': 'Medium', 'roles': 8},
            'Cybersecurity': {'avg_salary': 130000, 'min_salary': 95000, 'max_salary': 175000, 'growth': 4.0, 'demand': 'High', 'roles': 12},
            'Design_Mobile': {'avg_salary': 115000, 'min_salary': 80000, 'max_salary': 160000, 'growth': 3.2, 'demand': 'Medium', 'roles': 8}
        }
        
        # Convert to DataFrame
        market_df = pd.DataFrame([
            {
                'Category': category.replace('_', ' '),
                'Avg Salary': data['avg_salary'],
                'Min Salary': data['min_salary'],
                'Max Salary': data['max_salary'],
                'Growth Score': data['growth'],
                'Market Demand': data['demand'],
                'Available Roles': data['roles'],
                'Salary Range': data['max_salary'] - data['min_salary']
            }
            for category, data in categories_data.items()
        ])
        
        # Salary vs Growth Bubble Chart
        st.subheader("💰 Salary vs Growth Potential")
        fig_bubble = px.scatter(
            market_df,
            x='Growth Score',
            y='Avg Salary',
            size='Available Roles',
            color='Market Demand',
            hover_name='Category',
            hover_data=['Min Salary', 'Max Salary'],
            title='Career Categories: Salary vs Growth (Bubble Size = Available Roles)',
            color_discrete_map={'Very High': '#FF6B6B', 'High': '#4ECDC4', 'Medium': '#45B7D1'},
            size_max=60
        )
        fig_bubble.update_layout(height=500)
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Salary Range Comparison
        st.subheader("📊 Salary Range Analysis")
        
        # Create subplot with salary ranges
        fig_ranges = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48CAE4', '#DDA0DD']
        
        for i, (_, row) in enumerate(market_df.iterrows()):
            fig_ranges.add_trace(go.Scatter(
                x=[row['Min Salary'], row['Max Salary']],
                y=[row['Category'], row['Category']],
                mode='lines+markers',
                line=dict(width=8, color=colors[i % len(colors)]),
                marker=dict(size=12, color=colors[i % len(colors)]),
                name=row['Category'],
                hovertemplate=f"<b>{row['Category']}</b><br>" +
                             f"Range: ${row['Min Salary']:,} - ${row['Max Salary']:,}<br>" +
                             f"Average: ${row['Avg Salary']:,}<br>" +
                             f"Growth: {row['Growth Score']}/5.0<extra></extra>",
                showlegend=False
            ))
            
            # Add average salary markers
            fig_ranges.add_trace(go.Scatter(
                x=[row['Avg Salary']],
                y=[row['Category']],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='diamond', line=dict(width=2, color='black')),
                name=f"Avg: {row['Category']}",
                hovertemplate=f"<b>{row['Category']} Average</b><br>" +
                             f"${row['Avg Salary']:,}<extra></extra>",
                showlegend=False
            ))
        
        fig_ranges.update_layout(
            title='Salary Ranges by Career Category (Gold Diamonds = Average)',
            xaxis_title='Annual Salary (USD)',
            yaxis_title='Career Category',
            height=400,
            xaxis=dict(tickformat='$,.0f')
        )
        st.plotly_chart(fig_ranges, use_container_width=True)
        
        # Growth vs Demand Heatmap
        st.subheader("🔥 Growth & Demand Heatmap")
        
        # Create heatmap data
        demand_map = {'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1}
        heatmap_data = market_df.copy()
        heatmap_data['Demand Score'] = heatmap_data['Market Demand'].map(demand_map)
        
        fig_heatmap = px.density_heatmap(
            heatmap_data,
            x='Growth Score',
            y='Demand Score',
            z='Avg Salary',
            title='Career Attractiveness: Growth vs Market Demand (Color = Avg Salary)',
            labels={'Growth Score': 'Growth Potential (1-5)', 'Demand Score': 'Market Demand (1-5)'},
            color_continuous_scale='Viridis'
        )
        
        # Add scatter overlay
        fig_heatmap.add_trace(
            go.Scatter(
                x=heatmap_data['Growth Score'],
                y=heatmap_data['Demand Score'],
                mode='markers+text',
                text=heatmap_data['Category'],
                textposition='middle center',
                marker=dict(size=20, color='white', line=dict(width=2, color='black')),
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>' +
                             'Growth: %{x}/5.0<br>' +
                             'Demand: %{y}/5.0<extra></extra>'
            )
        )
        
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Market Summary Table
        st.subheader("📋 Comprehensive Market Data")
        
        # Format the dataframe for better display
        display_df = market_df.copy()
        display_df['Avg Salary'] = display_df['Avg Salary'].apply(lambda x: f"₹{x*USD_TO_INR/100000:.1f} LPA")
        display_df['Min Salary'] = display_df['Min Salary'].apply(lambda x: f"₹{x*USD_TO_INR/100000:.1f} LPA")
        display_df['Max Salary'] = display_df['Max Salary'].apply(lambda x: f"₹{x*USD_TO_INR/100000:.1f} LPA")
        display_df['Growth Score'] = display_df['Growth Score'].apply(lambda x: f"{x}/5.0")
        
        # Add trend indicators
        display_df['Trend'] = display_df['Growth Score'].apply(
            lambda x: "🚀" if float(x.split('/')[0]) >= 4.5 else 
                     "📈" if float(x.split('/')[0]) >= 4.0 else 
                     "➡️" if float(x.split('/')[0]) >= 3.5 else "📉"
        )
        
        # Reorder columns
        display_df = display_df[['Category', 'Trend', 'Avg Salary', 'Min Salary', 'Max Salary', 
                               'Growth Score', 'Market Demand', 'Available Roles']]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                'Category': st.column_config.TextColumn('Career Category', width='medium'),
                'Trend': st.column_config.TextColumn('Trend', width='small'),
                'Avg Salary': st.column_config.TextColumn('Average Salary', width='medium'),
                'Min Salary': st.column_config.TextColumn('Min Salary', width='medium'),
                'Max Salary': st.column_config.TextColumn('Max Salary', width='medium'),
                'Growth Score': st.column_config.TextColumn('Growth Score', width='small'),
                'Market Demand': st.column_config.SelectboxColumn('Market Demand', width='small'),
                'Available Roles': st.column_config.NumberColumn('Available Roles', width='small')
            }
        )
        
        # Market insights based on user profile
        if selected_skills:
            st.subheader("🎯 Personalized Market Insights")
            
            # Analyze user's skill alignment with market categories
            skill_category_relevance = {}
            for category in categories_data.keys():
                relevance_score = 0
                category_keywords = {
                    'AI_ML': ['python', 'machine learning', 'data science', 'tensorflow', 'deep learning'],
                    'Cloud_Infrastructure': ['aws', 'azure', 'docker', 'kubernetes', 'cloud'],
                    'Software_Engineering': ['python', 'java', 'javascript', 'react', 'node.js', 'programming'],
                    'Data_Engineering': ['sql', 'python', 'etl', 'data warehouse', 'big data'],
                    'Product_Management': ['product management', 'agile', 'scrum', 'strategy'],
                    'Cybersecurity': ['security', 'cyber', 'penetration testing', 'compliance'],
                    'Design_Mobile': ['design', 'ui', 'ux', 'mobile', 'react']
                }
                
                user_skills_lower = [skill.lower() for skill in selected_skills]
                category_skills = category_keywords.get(category, [])
                
                for user_skill in user_skills_lower:
                    for cat_skill in category_skills:
                        if user_skill in cat_skill or cat_skill in user_skill:
                            relevance_score += 1
                
                skill_category_relevance[category] = relevance_score
            
            # Show top matching categories
            top_matches = sorted(skill_category_relevance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_matches[0][1] > 0:
                col1, col2, col3 = st.columns(3)
                
                for i, (category, score) in enumerate(top_matches):
                    if score > 0:
                        with [col1, col2, col3][i]:
                            cat_data = categories_data[category]
                            st.info(f"""
                            **{category.replace('_', ' ')}**
                            
                            Skill Match: {score} skills
                            
                            Avg Salary: ₹{cat_data['avg_salary']*USD_TO_INR/100000:.1f} LPA
                            
                            Growth: {cat_data['growth']}/5.0
                            
                            Demand: {cat_data['demand']}
                            """)
            else:
                st.warning("💡 Consider developing skills in high-growth areas like AI/ML or Cloud Infrastructure!")
        
    except Exception as e:
        logger.error(f"Failed to generate market insights: {e}")
        st.error("Failed to load market insights. Please try again.")

with tab3:
    st.header("📈 Career Gap Analysis")
    
    if selected_skills and experience_years is not None:
        st.subheader("🎯 Your Career Profile Analysis")
        
        # User profile assessment
        col1, col2 = st.columns(2)
        
        with col1:
            # Current profile strength assessment
            profile_strength = len(selected_skills) * 10 + experience_years * 5
            max_strength = 100
            strength_percentage = min(profile_strength / max_strength * 100, 100)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = strength_percentage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Profile Strength"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Experience level breakdown
            exp_categories = {
                'Entry Level (0-2 years)': max(0, 3 - experience_years),
                'Mid Level (3-7 years)': max(0, min(5, experience_years - 2)),
                'Senior Level (8+ years)': max(0, experience_years - 7)
            }
            
            fig_exp = px.pie(
                values=list(exp_categories.values()),
                names=list(exp_categories.keys()),
                title="Experience Distribution",
                color_discrete_sequence=['#ff7f7f', '#7fbf7f', '#7f7fff']
            )
            fig_exp.update_layout(height=300)
            st.plotly_chart(fig_exp, use_container_width=True)
        
        # Skill gap analysis
        if st.button("🔍 Analyze Career Gaps", type="primary"):
            with st.spinner("Performing comprehensive career gap analysis..."):
                try:
                    # Get recommendations to understand target roles
                    recommendations = engine.get_recommendations_from_profile(user_profile, num_recommendations=10)
                    
                    if recommendations:
                        target_rec = recommendations[0]  # Primary recommendation
                        
                        st.subheader(f"🎯 Gap Analysis for {target_rec.role}")
                        
                        # Skill gap visualization
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Create skill comparison chart
                            all_skills = list(set(selected_skills + target_rec.required_skills))
                            skill_data = []
                            
                            for skill in all_skills:
                                has_skill = 1 if skill in selected_skills else 0
                                needs_skill = 1 if skill in target_rec.required_skills else 0
                                gap = needs_skill - has_skill
                                
                                skill_data.append({
                                    'Skill': skill,
                                    'Current Level': has_skill,
                                    'Required Level': needs_skill,
                                    'Gap': gap,
                                    'Status': 'Have' if gap == 0 and has_skill == 1 else 
                                             'Need to Learn' if gap == 1 else 
                                             'Optional' if gap == -1 else 'Not Required'
                                })
                            
                            skill_df = pd.DataFrame(skill_data)
                            
                            # Skill gap bar chart
                            fig_skills = go.Figure()
                            
                            # Current skills
                            fig_skills.add_trace(go.Bar(
                                name='Your Skills',
                                x=skill_df['Skill'],
                                y=skill_df['Current Level'],
                                marker_color='lightblue',
                                opacity=0.7
                            ))
                            
                            # Required skills
                            fig_skills.add_trace(go.Bar(
                                name='Required Skills',
                                x=skill_df['Skill'],
                                y=skill_df['Required Level'],
                                marker_color='orange',
                                opacity=0.7
                            ))
                            
                            fig_skills.update_layout(
                                title=f'Skill Gap Analysis for {target_rec.role}',
                                xaxis_title='Skills',
                                yaxis_title='Proficiency (0=None, 1=Required)',
                                barmode='group',
                                height=400,
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig_skills, use_container_width=True)
                        
                        with col2:
                            # Gap summary metrics
                            skills_you_have = len([s for s in skill_data if s['Status'] == 'Have'])
                            skills_to_learn = len([s for s in skill_data if s['Status'] == 'Need to Learn'])
                            skill_match_rate = (skills_you_have / len(target_rec.required_skills) * 100) if target_rec.required_skills else 100
                            
                            st.metric("Skills Match", f"{skill_match_rate:.0f}%", f"{skills_you_have}/{len(target_rec.required_skills)} skills")
                            st.metric("Skills to Learn", skills_to_learn)
                            salary_diff_inr = (target_rec.avg_salary - salary_expectation) * USD_TO_INR
                            st.metric("Target Salary", format_inr_salary(target_rec.avg_salary), f"+₹{salary_diff_inr/100000:.1f} LPA")
                            
                            # Experience gap
                            target_exp_map = {0: "0-2 years", 1: "3-7 years", 2: "8+ years"}
                            current_exp_level = min(experience_years // 3, 2)
                            
                            if current_exp_level < 2:
                                exp_gap = (2 - current_exp_level) * 3
                                st.metric("Experience Gap", f"{exp_gap} years", "to senior level")
                        
                        # Development timeline
                        st.subheader("📅 Development Timeline")
                        
                        # Create timeline based on skills to learn and experience
                        timeline_data = []
                        
                        # Skills development phases
                        skills_to_develop = [s['Skill'] for s in skill_data if s['Status'] == 'Need to Learn']
                        
                        if skills_to_develop:
                            # Phase 1: Foundation skills (0-6 months)
                            foundation_skills = skills_to_develop[:3]
                            if foundation_skills:
                                timeline_data.append({
                                    'Phase': 'Foundation Building',
                                    'Duration': '0-6 months',
                                    'Activities': f"Learn {', '.join(foundation_skills)}",
                                    'Milestone': 'Basic proficiency achieved'
                                })
                            
                            # Phase 2: Advanced skills (6-12 months)
                            advanced_skills = skills_to_develop[3:6]
                            if advanced_skills:
                                timeline_data.append({
                                    'Phase': 'Advanced Development',
                                    'Duration': '6-12 months',
                                    'Activities': f"Master {', '.join(advanced_skills)}",
                                    'Milestone': 'Job-ready competency'
                                })
                            
                            # Phase 3: Specialization (12-18 months)
                            timeline_data.append({
                                'Phase': 'Specialization',
                                'Duration': '12-18 months',
                                'Activities': 'Build portfolio, gain experience, networking',
                                'Milestone': f'Ready for {target_rec.role} positions'
                            })
                        
                        if timeline_data:
                            timeline_df = pd.DataFrame(timeline_data)
                            
                            # Timeline Gantt chart using horizontal bar chart
                            fig_timeline = go.Figure()
                            
                            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                            start_months = [0, 6, 12]
                            durations = [6, 6, 6]
                            
                            for i, (_, row) in enumerate(timeline_df.iterrows()):
                                fig_timeline.add_trace(go.Bar(
                                    name=row['Phase'],
                                    y=[row['Phase']],
                                    x=[durations[i]],
                                    base=[start_months[i]],
                                    orientation='h',
                                    marker_color=colors[i],
                                    text=f"{row['Duration']}",
                                    textposition='inside',
                                    hovertemplate=f"<b>{row['Phase']}</b><br>" +
                                                 f"Duration: {row['Duration']}<br>" +
                                                 f"Activities: {row['Activities']}<br>" +
                                                 f"Milestone: {row['Milestone']}<extra></extra>"
                                ))
                            
                            fig_timeline.update_layout(
                                title='Career Development Timeline (Months)',
                                xaxis_title='Timeline (Months)',
                                yaxis_title='Development Phases',
                                height=300,
                                showlegend=False,
                                barmode='stack'
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                            
                            # Detailed timeline table
                            st.dataframe(
                                timeline_df,
                                use_container_width=True,
                                column_config={
                                    'Phase': st.column_config.TextColumn('Development Phase', width='medium'),
                                    'Duration': st.column_config.TextColumn('Timeline', width='small'),
                                    'Activities': st.column_config.TextColumn('Key Activities', width='large'),
                                    'Milestone': st.column_config.TextColumn('Success Milestone', width='medium')
                                }
                            )
                        
                        # Action plan
                        st.subheader("🚀 Personalized Action Plan")
                        
                        action_tabs = st.tabs(["🎯 Immediate Actions", "📚 Learning Path", "💼 Experience Building"])
                        
                        with action_tabs[0]:
                            immediate_actions = [
                                f"✅ Set up learning environment for {skills_to_develop[0] if skills_to_develop else 'your target skills'}",
                                "📊 Create a portfolio showcasing current projects",
                                "🔗 Update LinkedIn profile with target role keywords",
                                "📚 Enroll in online courses for skill gaps",
                                "👥 Join professional communities in your target field"
                            ]
                            
                            for action in immediate_actions:
                                st.markdown(f"- {action}")
                        
                        with action_tabs[1]:
                            learning_resources = {
                                'AI_ML': ['Coursera ML Course', 'Kaggle Learn', 'Fast.ai', 'DeepLearning.ai'],
                                'Cloud_Infrastructure': ['AWS Training', 'Azure Fundamentals', 'Docker Documentation', 'Kubernetes.io'],
                                'Software_Engineering': ['FreeCodeCamp', 'LeetCode', 'System Design Primer', 'Clean Code'],
                                'Data_Engineering': ['DataCamp', 'Udacity Data Engineering', 'Apache Spark Docs', 'SQL Tutorial']
                            }
                            
                            category = target_rec.category
                            resources = learning_resources.get(category, ['Industry-specific courses', 'Professional certifications', 'Online tutorials', 'Practice projects'])
                            
                            st.markdown("**Recommended Learning Resources:**")
                            for resource in resources:
                                st.markdown(f"- 📖 {resource}")
                        
                        with action_tabs[2]:
                            experience_building = [
                                "🏗️ Contribute to open-source projects",
                                "💼 Seek volunteer opportunities using target skills",
                                "🤝 Find a mentor in your target field",
                                "🗣️ Attend industry meetups and conferences",
                                "📝 Write technical blogs about your learning journey",
                                "🎯 Apply for internships or entry-level positions"
                            ]
                            
                            for exp in experience_building:
                                st.markdown(f"- {exp}")
                        
                        # Progress tracking
                        st.subheader("📊 Track Your Progress")
                        
                        progress_col1, progress_col2 = st.columns(2)
                        
                        with progress_col1:
                            st.markdown("**Skills Development Checklist:**")
                            for skill in skills_to_develop[:5]:
                                st.checkbox(f"Learn {skill}", key=f"skill_{skill}")
                        
                        with progress_col2:
                            st.markdown("**Monthly Milestones:**")
                            milestones = [
                                "Complete first online course",
                                "Build first project",
                                "Get mentor feedback",
                                "Apply to first role",
                                "Complete portfolio"
                            ]
                            for i, milestone in enumerate(milestones):
                                st.checkbox(milestone, key=f"milestone_{i}")
                    
                    else:
                        st.warning("Unable to generate gap analysis. Please ensure your profile is complete.")
                
                except Exception as e:
                    logger.error(f"Failed to analyze career gaps: {e}")
                    st.error("Failed to analyze career gaps. Please try again.")
    
    else:
        st.info("👈 Please select your skills and experience level in the sidebar to get personalized gap analysis!")

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