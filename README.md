# 🌿 CropSense AI — Crop Disease Detection

Real-time leaf disease detection using a 5-layer AI fallback pipeline.

## 🚀 Deploy to Render (FREE)

### Step 1 — Push to GitHub (no secrets needed)

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cropsense-ai.git
git push -u origin main --force
```

> **Why this works now:** `.env` is in `.gitignore` so secrets are never pushed.
> `render.yaml` uses `sync: false` which means you enter values in Render's dashboard.

---

### Step 2 — Deploy on Render

1. Go to **https://render.com** → sign in
2. Click **New +** → **Web Service**
3. Connect your GitHub repo → select **cropsense-ai**
4. Render auto-detects `render.yaml` and creates the service
5. You'll see **Environment Variables** in the dashboard — add these 5 keys:

| Key | Value |
|-----|-------|
| `ROBOFLOW_KEY` | *(your key)* |
| `HF_TOKEN` | *(your key)* |
| `DEEPAI_KEY` | *(your key)* |
| `GROQ_KEY` | *(your key)* |
| `OPENAI_KEY` | *(your key)* |

6. Click **Save Changes** → **Manual Deploy** → **Deploy latest commit**
7. Wait ~3 minutes → your live URL appears at the top

---

## 💻 Run Locally

```bash
npm install
npm start
# Open http://localhost:3000
```

The `.env` file has all keys pre-filled for local use.

---

## Architecture

```
Image Upload (base64 JSON)
    │
    ├─ L1: Roboflow      (custom leaf model)
    ├─ L2: HuggingFace   (PlantVillage 38-class MobileNetV2)
    ├─ L3: DeepAI        (image classifier fallback)
    ├─ L4: Groq LLaMA3   (treatment + fertilizer advice)
    └─ L5: OpenAI GPT-4o (final vision fallback)
```

## API Endpoints

- `GET  /api/health`  — check key status
- `POST /api/detect`  — `{ imageBase64: "..." }` → full result JSON
- `POST /api/chat`    — `{ message: "...", lastDisease: {...} }` → reply
