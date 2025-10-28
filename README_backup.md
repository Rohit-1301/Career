# 🚀 CareerSaathi – AI-Powered Career Intelligence Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Google AI](https://img.shields.io/badge/Google%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google/)

**CareerSaathi** is a comprehensive AI-powered career intelligence platform that combines advanced machine learning with personalized coaching to help professionals navigate their career journey. From skill-based career recommendations to market insights, CareerSaathi provides data-driven suggestions tailored to each user's unique profile with real Indian salary data.

---

## ✨ Key Features

### 🎯 **Skill-Based Career Recommendations**
- **Advanced Decision Tree Algorithm**: ML model trained on 100+ tech career roles with aggressive skill matching
- **Zero-Base Scoring System**: Precision skill-to-career matching with 90% boost for exact matches
- **10 Diverse Recommendations**: Get 10 personalized career paths across multiple categories
- **Real-time Analysis**: Dynamic recommendations that adapt instantly as you update your skills

### � **Indian Salary Intelligence**
- **INR Currency Display**: All salaries shown in ₹ Lakhs Per Annum (LPA) and Crores
- **Comprehensive Tech Salary Data**: 100+ roles with entry, mid-level, and senior salary ranges
- **Growth Projections**: Career growth outlook percentages for each role
- **Interactive Comparisons**: Visual salary range comparisons and career progression charts

### � **Smart Skill Matching**
- **50+ Skill Keywords**: Comprehensive mapping covering Web Dev, AI/ML, Cloud, Data, and more
- **Category Intelligence**: Automatic categorization (Software Engineering, AI/ML, Cloud Infrastructure, etc.)
- **Precision Matching**: JavaScript/HTML → Software Engineering, Python/ML → AI/ML, AWS/Docker → Cloud roles
- **Multi-Skill Analysis**: Stronger recommendations when multiple related skills are selected

### 🤖 **Intelligent AI Coaching**
- **Contextual Conversations**: LangChain-powered AI that remembers your career journey
- **Document Analysis**: Upload resumes and job descriptions for personalized insights
- **Profile-Based Recommendations**: AI considers your complete skill profile for better suggestions

### 🔐 **Secure & Persistent**
- **Firebase Authentication**: Secure user management with profile persistence
- **Cloud Storage**: Document uploads stored securely in Firebase Storage
- **Conversation History**: Long-term memory with Firestore for continuous learning

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐ │
│  │  Chat   │  │Documents │  │  History   │  │  Career   │ │
│  │ Page    │  │  Page    │  │   Page     │  │ Insights  │ │
│  └────┬────┘  └────┬─────┘  └─────┬──────┘  └─────┬─────┘ │
└───────┼───────────┼──────────────┼────────────────┼────────┘
        │           │              │                │
        ├───────────┴──────────────┼────────────────┤
        │                          │                │
        ▼                          ▼                ▼
┌──────────────────┐      ┌─────────────┐  ┌────────────────────┐
│  LangChain       │      │  Firebase   │  │  Career Decision   │
│  Pipeline        │      │  Services   │  │  Tree Engine       │
│  ┌────────────┐  │      │ ┌─────────┐ │  │  ┌──────────────┐  │
│  │ Gemini AI  │  │      │ │  Auth   │ │  │  │ Skill-Based  │  │
│  └────────────┘  │      │ └─────────┘ │  │  │ Matching     │  │
│  ┌────────────┐  │      │ ┌─────────┐ │  │  └──────────────┘  │
│  │ Chat       │  │      │ │Firestore│ │  │  ┌──────────────┐  │
│  │ Memory     │  │      │ └─────────┘ │  │  │ Zero-Base    │  │
│  └────────────┘  │      │ ┌─────────┐ │  │  │ Scoring      │  │
└──────────────────┘      │ │ Storage │ │  │  └──────────────┘  │
                          │ └─────────┘ │  │  ┌──────────────┐  │
                          └─────────────┘  │  │ 10 Diverse   │  │
                                           │  │ Suggestions  │  │
                                           │  └──────────────┘  │
                                           └────────────────────┘
                                                    │
                                                    ▼
                                           ┌────────────────────┐
                                           │    tech.csv        │
                                           │  (100+ Roles)      │
                                           │  INR Salary Data   │
                                           └────────────────────┘
```

### 🧩 **Core Components**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Multi-page web application with interactive UI |
| **AI Engine** | LangChain + Google Gemini | Conversational AI with context awareness |
| **ML Pipeline** | Scikit-learn | Decision tree models with aggressive skill matching |
| **Skill Matcher** | Custom Algorithm | Zero-base scoring with 90% boost for exact matches |
| **Data Viz** | Plotly + Pandas | Interactive INR salary charts and comparisons |
| **Backend** | Firebase | Authentication, database, and file storage |
| **Data Source** | tech.csv | 100+ tech roles with LPA salary ranges |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Firebase project with Authentication, Firestore, and Storage enabled
- Google AI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rohit-1301/Career.git
   cd careersaathi
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your credentials
   GOOGLE_API_KEY=your_google_ai_api_key
   ```

5. **Firebase Configuration**
   - Place your Firebase service account JSON file in the project root
   - Update the path in your environment configuration

6. **Run the application**
   ```bash
   streamlit run streamlit_app\app.py
   ```

---

## 📁 Project Structure

```
careersaathi/
├── 📱 streamlit_app/           # Frontend application
│   ├── app.py                  # Main app entry point with auth
│   ├── pages/                  # Multi-page application
│   │   ├── 1_💬_Chat.py       # AI chat interface
│   │   ├── 2_📄_Documents.py  # Document management
│   │   ├── 3_📊_History.py    # Conversation history
│   │   └── 4_🎯_Career_Insights.py  # Career recommendations
│   └── components/             # Reusable UI components
│       ├── auth.py            # Authentication components
│       ├── chat.py            # Chat interface components
│       └── session.py         # Session management
├── 🤖 ai/                      # AI and ML components
│   ├── career_insights.py      # Decision tree with skill matching
│   ├── career_recommendations.py  # 10 diverse recommendations
│   └── langchain_pipeline.py   # LangChain + Gemini integration
├── 🔥 firebase/               # Firebase integrations
│   ├── auth.py                # Authentication services
│   ├── config.py              # Firebase initialization
│   ├── db.py                  # Firestore operations
│   └── storage.py             # File storage management
├── ⚙️ services/               # Core business logic
│   └── utils.py               # Shared utilities
├── 📊 tech.csv                # 100+ tech roles with INR salary data
├── 🤖 career_model.joblib     # Trained ML model (auto-generated)
├── 📋 requirements.txt        # Python dependencies
├── 🔧 .env.example           # Environment template
└── 🧪 test_skill_fix.py      # Skill matching validation tests
```

---

## 🎯 Core Features Deep Dive

### **1. Skill-Based Career Matching Algorithm**
```python
# Advanced Zero-Base Scoring System
- Input: User-selected skills (JavaScript, React, Python, etc.)
- Processing: 
  * Maps skills to categories (Web Dev → Software_Engineering)
  * Starts with zero base scores for all categories
  * Applies 90% boost for exact skill matches
  * Applies 70% boost for partial matches
  * Multiplies by 1 + (0.5 × skill_count) for multiple skills
  * Penalizes non-matching categories by 90%
- Output: 10 diverse career recommendations with confidence scores
- Accuracy: 100% confidence for exact skill matches
```

**Skill Coverage:**
- **Web Development**: JavaScript, JS, React, Angular, Vue, Node.js, HTML, CSS, Frontend, Backend, Full-Stack, API
- **AI/ML**: Python, Machine Learning, ML, Deep Learning, DL, TensorFlow, PyTorch, Statistics, Data Science
- **Cloud & Infrastructure**: AWS, Azure, GCP, Docker, Kubernetes, DevOps, CI/CD, Cloud
- **Data Engineering**: SQL, NoSQL, ETL, Data Pipeline, Spark, Hadoop, Database
- **Mobile Development**: iOS, Android, Swift, Kotlin, React Native, Flutter
- **Design**: UI, UX, Figma, Adobe, Design, Mobile

### **2. Indian Salary Intelligence**
- **Currency Format**: All salaries displayed in ₹ Lakhs Per Annum (LPA)
- **Conversion Rate**: 1 USD = ₹83 INR
- **Salary Ranges**: 
  * Entry Level: ₹3-7 LPA (Software roles)
  * Mid Level: ₹8-12 LPA (Software roles)
  * Senior Level: ₹18-30 LPA (Software roles)
- **Display Format**: 
  * Under 1 Crore: "₹12.45 LPA"
  * Over 1 Crore: "₹1.25 Cr"
- **Growth Projections**: Percentage ranges showing career growth outlook (e.g., "10-35%")

### **3. 10 Diverse Career Recommendations**
- **Primary Category**: Top 3 roles from your best-matching category
- **Alternative Categories**: 2 roles each from 3-4 related categories
- **Diversity Algorithm**: Ensures recommendations span multiple career paths
- **Real-time Updates**: Auto-generates as you select skills
- **Confidence Scores**: Shows match percentage for each recommendation

### **4. Interactive Career Insights Dashboard**
- **Skill Selection**: Multi-select dropdown with popular tech skills
- **Experience Level**: Slider from 0-20 years
- **Salary Expectations**: ₹3 LPA to ₹1 Cr range (default ₹12 LPA)
- **Growth Preferences**: Moderate/High/Exponential options
- **Visual Analytics**:
  * Salary range comparison charts (Plotly bubble charts)
  * Career growth heatmaps
  * Skill gap analysis
  * Required skills breakdown

### **5. AI-Powered Career Coaching**
- **Contextual Memory**: Remembers your career journey and preferences
- **Document Intelligence**: Analyzes resumes and job descriptions
- **Adaptive Responses**: Learns from your interactions
- **Profile Integration**: Uses your skill profile for personalized advice

---

## 🔧 Configuration

### **Environment Variables**
```bash
# Google AI Configuration
GOOGLE_API_KEY=your_google_ai_api_key

# Firebase Configuration
FIREBASE_TYPE=service_account
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_PRIVATE_KEY_ID=your_private_key_id
FIREBASE_PRIVATE_KEY=your_private_key
FIREBASE_CLIENT_EMAIL=your_client_email
FIREBASE_CLIENT_ID=your_client_id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### **Firebase Setup**
1. Create a new Firebase project at [Firebase Console](https://console.firebase.google.com/)
2. Enable Authentication (Email/Password)
3. Create Firestore database
4. Enable Storage
5. Generate service account credentials
6. Download and place the JSON file in your project root

---

## 📊 Data Sources & Models

### **Career Data Pipeline**
- **Source**: `tech.csv` with 100+ tech career roles
- **Data Format**: 
  * JobTitle: Role name (e.g., "Frontend Developer")
  * Domain: Category (e.g., "Software Development")
  * EntrySalary_LPA: Entry-level salary in Lakhs Per Annum
  * MidSalary_LPA: Mid-level salary in LPA
  * SeniorSalary_LPA: Senior-level salary in LPA
  * GrowthOutlook_pct_range: Growth percentage (e.g., "10-35")
- **Coverage**: 
  * Software Development (Frontend, Backend, Full-Stack, Mobile, etc.)
  * Data & ML (Data Scientist, ML Engineer, AI Researcher, etc.)
  * Cloud & Infrastructure (DevOps, Cloud Architect, Platform Engineer, etc.)
  * Leadership & Management roles
- **Processing**: Salary parsing, growth score extraction, skill estimation

### **Machine Learning Models**
- **Algorithm**: Decision Tree Classifier with aggressive skill-based adjustment
- **Features**: 
  * Experience level (0-20 years)
  * Estimated skill count (based on role complexity)
  * Salary expectations (in LPA)
  * Growth preferences (1-3 scale)
- **Training**: 
  * 99 career records from tech.csv
  * Stratified sampling by category
  * Model accuracy: ~50% (baseline before skill adjustment)
- **Skill Matching Enhancement**:
  * Zero-base scoring system
  * 90% boost for exact skill-category matches
  * 70% boost for partial matches
  * 50% multiplier per additional matching skill
  * 90% penalty for non-matching categories
- **Deployment**: 
  * Saved as `career_model.joblib`
  * Auto-retrains if model file is missing
  * Loads on first use for fast predictions

---

## 🧪 Testing & Quality Assurance

### **Test Suite**
```bash
# Test skill matching algorithm
python test_skill_fix.py

# Expected output:
# ✅ Test 1: JavaScript/HTML/React → Software_Engineering (100%)
# ✅ Test 3: Python/ML/TensorFlow → AI_ML (33%)
```

### **Test Coverage**
- ✅ Decision tree training and prediction
- ✅ Skill-to-category mapping accuracy
- ✅ Zero-base scoring system validation
- ✅ 10 diverse recommendations generation
- ✅ INR currency conversion and display
- ✅ User profile extraction and processing

### **Known Test Results**
```
Test Case 1: Web Development Skills
- Input: JavaScript, HTML/CSS, React, Node.js
- Expected: Software_Engineering
- Result: ✅ 100% confidence match
- Top Roles: Director of Engineering, Game Engine Engineer, Chip Verification Engineer

Test Case 2: UI/UX Design Skills
- Input: UI, UX, Design, Mobile
- Expected: Design_Mobile category
- Result: ⚠️ Leadership (100%) - No UI/UX roles in tech.csv dataset
- Note: Dataset focuses on technical roles; design roles not included

Test Case 3: AI/ML Skills
- Input: Python, ML, TensorFlow, DL
- Expected: AI_ML
- Result: ✅ 33% confidence match
- Top Role: AI Researcher
```

---

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run streamlit_app\app.py
```

### **Streamlit Cloud**
1. Connect your GitHub repository
2. Set environment variables in Streamlit Cloud dashboard:
   - `GOOGLE_API_KEY`
   - `FIREBASE_API_KEY`
   - `FIREBASE_AUTH_DOMAIN`
   - `FIREBASE_PROJECT_ID`
   - `FIREBASE_STORAGE_BUCKET`
   - `GOOGLE_APPLICATION_CREDENTIALS` (as secrets.toml)
3. Deploy with automatic builds

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/main.py"]
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for significant changes
- Ensure all tests pass before submitting

---

## 📈 Future Roadmap

### **Upcoming Features**
- � **UI/UX Design Roles**: Add design-focused career paths to tech.csv
- 🔄 **Skill Learning Paths**: Personalized roadmaps to acquire required skills
- 📊 **Career Comparison Tool**: Side-by-side role comparisons with salary bands
- 🌐 **Job Market Integration**: Real-time job posting analysis from major portals
- 📱 **Mobile Optimization**: Better responsive design for mobile devices
- 🎓 **Certification Tracking**: Integration with Coursera, Udemy, etc.

### **Technical Improvements**
- ⚡ **Caching Optimization**: Redis/Streamlit cache for faster predictions
- 📈 **More ML Models**: Random Forest, XGBoost for ensemble predictions
- � **Semantic Skill Search**: Embeddings-based skill similarity matching
- 📊 **Historical Trend Analysis**: Track salary trends over time
- 🔐 **Enhanced Security**: OAuth2 and role-based access control
- 🌍 **Multi-Region Data**: Support for global salary data (US, EU, Asia)

### **Data Expansion**
- 🎯 **Non-Tech Roles**: Healthcare, Finance, Marketing, Sales careers
- 📚 **Skill Requirements**: Detailed skill lists for each role
- 🏆 **Certification Data**: Required certifications and courses
- 📈 **Company Data**: Top companies for each role with hiring trends

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google AI**: For providing the Gemini API
- **Streamlit**: For the amazing web app framework
- **Firebase**: For backend infrastructure
- **LangChain**: For AI orchestration
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning capabilities

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Rohit-1301/Career/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Rohit-1301/Career/discussions)
- **Email**: career.saathi@gmail.com

---

<div align="center">

**Built with ❤️ by the CareerSaathi Team**

[⭐ Star this repo](https://github.com/Rohit-1301/Career) | [🐛 Report Bug](https://github.com/Rohit-1301/Career/issues) | [💡 Request Feature](https://github.com/Rohit-1301/Career/issues)

</div>
- **AI Layer:** LangChain + `langchain-google-genai` wrapping Gemini models.
- **Data Layer:** Firebase Auth for users, Firestore for history, Storage for uploads.
- **Support:** `services/utils.py` centralises logging and environment loading.

---

## 📁 Project Structure

```
careersaathi/
├── ai/
│   ├── gemini_client.py          # Gemini client wrapper
│   └── langchain_pipeline.py     # Conversation orchestration + memory
├── firebase/
│   ├── auth.py                   # Authentication helpers
│   ├── config.py                 # Shared Firebase initialisation logic
│   ├── db.py                     # Firestore utilities
│   ├── storage.py                # Firebase Storage helpers
│   └── functions/                # Reserved for Cloud Functions integrations
├── services/
│   └── utils.py                  # Logging + environment helpers
├── streamlit_app/
│   ├── app.py                    # Entry point with auth gate
│   ├── components/               # Streamlit widgets (auth, chat, session)
│   └── pages/                    # Dashboard, History, Uploads views
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ✅ Prerequisites

- Python **3.10+** (3.11 recommended)
- Firebase project with **Authentication**, **Firestore (Native)**, and **Storage** enabled
- Google Gemini API access (create a key in [Google AI Studio](https://aistudio.google.com/app/apikey))

---

## 🚀 Getting Started

### 1. Clone the repository
```powershell
git clone <your-fork-url> careersaathi
cd careersaathi
```

### 2. Create a virtual environment & install deps
```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

1. Copy `.env.example` ➜ `.env`.
2. Populate Firebase keys (from **Project settings → General → Your apps → Firebase SDK snippet**).
3. Download a **Firebase Admin SDK** JSON file and set `GOOGLE_APPLICATION_CREDENTIALS` to its absolute path.
4. Set `GEMINI_API_KEY` with the key from Google AI Studio.

> **Bucket naming tip:** Firebase’s default bucket follows `project-id.appspot.com`. Ensure `FIREBASE_STORAGE_BUCKET` matches the bucket shown in the Storage console (look for the `gs://` string).

### 4. Enable Firebase products

- **Authentication:** Turn on *Email/Password*.
- **Firestore:** Start in *Native mode*.
- **Storage:** Click **Get Started**, choose a location, then update the bucket rules to allow authenticated access:

```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /{allPaths=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

### 5. Run the Streamlit app
```powershell
streamlit run streamlit_app/app.py
```

### 6. Sign up & explore

- Create an account from the landing screen.
- Ask questions on **Dashboard**, upload files alongside prompts, and view inline AI responses.
- Revisit previous conversations via **History** and download/delete uploads from **Uploads**.

---

## ⚙️ Configuration Reference

| Variable | Description |
| --- | --- |
| `FIREBASE_API_KEY` | Web API key from Firebase project settings |
| `FIREBASE_AUTH_DOMAIN` | Typically `<project-id>.firebaseapp.com` |
| `FIREBASE_DATABASE_URL` | Optional; used if Realtime Database is enabled |
| `FIREBASE_PROJECT_ID` | Your Firebase project ID |
| `FIREBASE_STORAGE_BUCKET` | Storage bucket name (e.g., `project-id.appspot.com`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Absolute path to Firebase Admin SDK service account JSON |
| `GEMINI_API_KEY` | Google AI Studio API key |
| `APP_SECRET_KEY` | Streamlit session secret |

Restart the Streamlit server whenever `.env` or credentials change.

---

## 🧪 Development & Validation

- **Type / syntax check:**
  ```powershell
  python -m compileall ai firebase services streamlit_app
  ```
- **Run unit tests (if added later):**
  ```powershell
  pytest
  ```
- To experiment with Gemini prompts, open a Python shell and interact with `ai/gemini_client.py` directly.

---

## 🧰 Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError` for local packages | `sys.path` not updated | Relaunch via `streamlit run streamlit_app/app.py` from project root |
| Firebase auth errors (`OPERATION_NOT_ALLOWED`) | Email/Password provider disabled | Enable provider in Firebase Authentication console |
| `ValueError: The default Firebase app does not exist` | Admin SDK not initialised | Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account JSON |
| `404 ... The specified bucket does not exist` | Wrong `FIREBASE_STORAGE_BUCKET` value | Copy the exact bucket name shown in Firebase Storage (`gs://…`) |
| Gemini `404 models/... not found` | Model unavailable for API key | Use `gemini-pro` or `gemini-2.0-flash-lite` (check [model availability](https://ai.google.dev/gemini-api/docs/models)) |

---

## 🛣️ Roadmap Ideas

- Add semantic retrieval for uploaded files (e.g., LangChain retrievers + embeddings).
- Introduce team spaces & role-based access controls.
- Deploy Cloud Functions for scheduled nudges or resume analysis jobs.
- Expand analytics with Streamlit charts powered by Firestore aggregates.

---

## 🤝 Contributing

Issues and pull requests are welcome. Please open an issue to discuss large feature ideas before starting work.

Happy coaching! ✨
