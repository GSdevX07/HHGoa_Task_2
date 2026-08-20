# 🚀 Online Deployment Guide — Voice-Enabled RAG System

This guide provides step-by-step instructions for deploying the **Voice-Enabled Multilingual RAG Engine** online across different free and cloud platforms.

---

## 📋 Required Environment Variables

Before deploying, make sure you have your API keys ready:

| Variable | Description | Required | Where to Get |
| :--- | :--- | :--- | :--- |
| `GROQ_API_KEY` | Groq LLM API Key (Ultra-fast generation) | **Yes** | [console.groq.com](https://console.groq.com) |
| `SARVAM_API_KEY` | Sarvam AI STT API Key (Indic speech-to-text) | **Yes** (for voice) | [console.sarvam.ai](https://console.sarvam.ai) |
| `ELEVENLABS_API_KEY` | ElevenLabs Scribe STT API Key | Optional (fallback) | [elevenlabs.io](https://elevenlabs.io) |
| `OPENAI_API_KEY` | OpenAI API Key | Optional (fallback) | [platform.openai.com](https://platform.openai.com) |
| `STT_PROVIDER` | Preferred STT provider (`sarvam` or `elevenlabs`) | Optional (default: `sarvam`) | - |
| `EMBEDDING_MODEL` | Embedding model for Dense retrieval | Optional (default: `all-MiniLM-L6-v2`) | - |
| `RERANKER_MODEL` | Cross-encoder reranker model | Optional (default: `BAAI/bge-reranker-v2-m3`) | - |
| `RERANKER_ENABLED` | Enable/disable reranker (`true`/`false`) | Optional (default: `true`) | - |

---

## 🌟 Option 1: Deploy to Hugging Face Spaces (Recommended — 100% Free 16GB RAM)

Hugging Face Spaces offers **free 16GB RAM + 2 vCPUs**, which is ideal for running Sentence Transformers, FAISS, and BM25 with zero memory constraints.

### Steps:
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set:
   - **Space name**: `voice-rag-msmarco`
   - **Space SDK**: `Docker` -> `Blank`
   - **Visibility**: `Public` (or `Private`)
3. Click **Create Space**.
4. In your Space settings, go to **Variables and secrets** -> **New secret**:
   - Add `GROQ_API_KEY`
   - Add `SARVAM_API_KEY`
   - Add `ELEVENLABS_API_KEY` (optional)
5. Push your repository to the Hugging Face Space git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/voice-rag-msmarco
   git push space main
   ```
6. Hugging Face will automatically build the `Dockerfile` and launch the web app on port 7860/8000!

---

## ⚡ Option 2: Deploy to Render (1-Click Web Service)

Render provides a generous free tier with automatic HTTPS and Git integration.

### Quick Deploy via Render Dashboard:
1. Push your code to your GitHub repository:
   ```bash
   git push origin main
   ```
2. Log into [Render.com](https://dashboard.render.com).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository: `GSdevX07/HHGoa_Task_2`.
5. Configure the service:
   - **Runtime**: `Docker` (recommended) or `Python`
   - **Branch**: `main`
   - **Region**: Closest to your users (e.g., `Singapore` or `Frankfurt`)
   - **Instance Type**: `Free`
6. In **Environment Variables**, add:
   - `GROQ_API_KEY` = `your_groq_key`
   - `SARVAM_API_KEY` = `your_sarvam_key`
   - `STT_PROVIDER` = `sarvam`
   - `RERANKER_ENABLED` = `true`
7. Click **Create Web Service**. Render will build and deploy your app.

---

## 🚂 Option 3: Deploy to Railway

Railway provides seamless Docker/Python deployment with instant CI/CD.

### Steps:
1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select `GSdevX07/HHGoa_Task_2`.
4. Railway will automatically detect the `Dockerfile` or `Procfile`.
5. Go to **Variables** tab in your Railway service and add:
   - `GROQ_API_KEY`
   - `SARVAM_API_KEY`
   - `PORT` (Railway provides this automatically)
6. Go to **Settings** -> **Networking** -> click **Generate Domain** to get your public live URL (e.g., `https://voice-rag-production.up.railway.app`).

---

## 🪂 Option 4: Deploy to Fly.io

Fly.io runs applications globally close to users on lightweight microVMs.

### Steps:
1. Install Fly CLI: [fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)
2. Log in:
   ```bash
   fly auth login
   ```
3. Launch the app (from project root):
   ```bash
   fly launch --no-deploy
   ```
4. Set your production secrets:
   ```bash
   fly secrets set GROQ_API_KEY="your_groq_key" SARVAM_API_KEY="your_sarvam_key"
   ```
5. Deploy:
   ```bash
   fly deploy
   ```

---

## 🐳 Option 5: Self-Hosted Cloud VM / Docker (AWS, GCP, DigitalOcean, Hetzner)

Deploy directly to any Linux VPS running Docker.

### Steps on your server:
1. Clone the repository:
   ```bash
   git clone https://github.com/GSdevX07/HHGoa_Task_2.git
   cd HHGoa_Task_2
   ```
2. Create `.env` file:
   ```bash
   cp backend/.env.example .env
   nano .env  # Enter your API keys
   ```
3. Run with Docker Compose:
   ```bash
   docker compose up --build -d
   ```
4. Your application is live at `http://your-server-ip:8000`.
5. (Optional) Set up Nginx + Certbot for SSL/HTTPS:
   ```nginx
   server {
       server_name yourdomain.com;
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

---

## 🔍 Verifying the Deployment

Once deployed, you can verify your service status:

- **Web UI**: Navigate to your deployed URL (e.g. `https://your-app.onrender.com`)
- **Health Check Endpoint**: `https://your-app.onrender.com/api/health`
- **Swagger API Docs**: `https://your-app.onrender.com/docs`
- **Retrieval Info**: `https://your-app.onrender.com/api/retrieval/info`
