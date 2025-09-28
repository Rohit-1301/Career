# CareerSaathi – AI Career Coach

CareerSaathi is a full-stack Streamlit application that pairs Google Gemini with LangChain and Firebase to deliver a personalised career coaching experience. Users sign in with Firebase Auth, chat with an AI assistant powered by Gemini, upload supporting documents, and revisit their history at any time.

---

## ✨ Highlights

- 🔐 **Secure onboarding** – Email/password authentication via Firebase Auth with profile persistence.
- 🧠 **Contextual Gemini coaching** – LangChain orchestrates prompts, memory, and model retries for robust responses.
- 📁 **Document-aware insights** – Resumes or job descriptions are stored in Firebase Storage and can be injected into AI context.
- 🗃️ **Long-term memory** – Firestore captures conversations, uploads, and metadata for future analysis.
- 🧩 **Modular architecture** – Clear layers for UI, AI pipeline, Firebase services, and shared utilities keep the codebase maintainable.

---

## 🛠️ Architecture Overview

```
┌────────────────┐      ┌────────────────────┐
│ Streamlit UI   │ ───▶ │ LangChain Pipeline │ ───▶ Google Gemini (chat)
└────────────────┘      └────────────────────┘
        │                         │
        ▼                         ▼
┌────────────────┐      ┌────────────────────┐
│ Firebase Auth  │      │ Firestore / Storage│
└────────────────┘      └────────────────────┘
```

- **Frontend:** Streamlit multipage app (`streamlit_app/`) with reusable components.
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
