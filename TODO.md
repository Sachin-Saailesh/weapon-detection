# 📋 Setup To-Do List

This document outlines the manual steps required to set up all platform connectors and API keys before running the weapon detection pipeline in a production environment.

## 1. 🔑 HyperAI Registry

_(Cost: Assumed free/internal for this project context)_

Used for tracking the initial dataset and pushing model versions.

- [ ] Go to [HyperAI](https://hyperai.example.com) (or your internal HyperAI instance).
- [ ] Log in and navigate to **Settings > API Tokens**.
- [ ] Generate a new Personal Access Token.
- [ ] Set the environment variable:
  ```bash
  export HYPERAI_TOKEN="your_hyperai_token_here"
  ```

## 2. 🗄️ AWS S3 (Storage)

_(Cost: AWS has a free tier for 12 months (5GB standard storage), but incurs costs beyond that. **Free Alternative**: Use local storage or Firebase Spark Plan.)_

Used for backing up the final ONNX model and metrics artifacts.

- [ ] Log in to the [AWS Management Console](https://console.aws.amazon.com).
- [ ] Go to **IAM (Identity and Access Management) > Users**.
- [ ] Select or create a user with `AmazonS3FullAccess` (or scoped bucket access).
- [ ] Go to the **Security credentials** tab and click **Create access key**.
- [ ] Save the Access Key ID and Secret Access Key.
- [ ] Set the environment variables:
  ```bash
  export AWS_ACCESS_KEY_ID="your_aws_access_key"
  export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
  ```

## 3. 🔥 Firebase (Storage Alternative)

_(Cost: **Completely Free** generous "Spark" plan (5GB storage, 1GB/day downloads), highly recommended as a free S3 alternative)._

Used if you prefer Firebase Storage for model backups over S3.

- [ ] Go to the [Firebase Console](https://console.firebase.google.com).
- [ ] Open your project and go to **Project setings > Service accounts**.
- [ ] Click **Generate new private key**.
- [ ] Download the generated JSON file.
- [ ] Place the JSON file securely on your machine (e.g., in a `.secrets/` folder).
- [ ] Set the environment variable to point to that file:
  ```bash
  export FIREBASE_CRED_PATH="/absolute/path/to/firebase-adminsdk.json"
  ```

## 4. 📈 Weights & Biases (Tracking)

_(Cost: **Completely Free** for personal projects (100GB storage). Perfect for this use case)._

Used for tracking YOLO training loss, mAP curves, and experiment logs.

- [ ] Go to [Weights & Biases (wandb.ai)](https://wandb.ai/).
- [ ] Sign up or log in.
- [ ] Navigate to your **Settings** page ([wandb.ai/settings](https://wandb.ai/settings)).
- [ ] Copy your API key from the **API keys** section.
- [ ] **To use this in Kaggle:**
  1. Open your Kaggle Notebook.
  2. Click **Add-ons -> Secrets** from the top menu.
  3. Click **Add a new secret**.
  4. Set the **Label** to `WANDB_API_KEY`.
  5. Paste your copied API key into the **Value** field and save.
  6. **Make sure the toggle next to the secret is turned ON (blue) so the notebook can access it.**

## 5. ⚡ Kaggle (Dataset Download & Training)

_(Cost: **Completely Free**. Provides 30 hours of free dual GPU (T4x2 or P100) training per week)._

Used if downloading or running pipelines natively on Kaggle.

- [ ] Go to [Kaggle](https://www.kaggle.com).
- [ ] Log in, click on your profile picture, and go to **Settings**.
- [ ] Scroll down to the **API** section and click **Create New Token**.
- [ ] This downloads a `kaggle.json` file containing your username and key.
- [ ] You can either place this file in `~/.kaggle/kaggle.json` OR set the environment variables manually:
  ```bash
  export KAGGLE_USERNAME="your_kaggle_username"
  export KAGGLE_KEY="your_kaggle_api_key"
  ```

---

## 🚀 Running the Notebook

If you are running the `weapon_detection_system.ipynb` notebook **locally**, make sure you start Jupyter with these variables loaded:

```bash
# Load variables
export HYPERAI_TOKEN="..."
export AWS_ACCESS_KEY_ID="..."
export WANDB_API_KEY="..."

# Start notebook
jupyter notebook weapon_detection_system.ipynb
```

If you are running on **Kaggle**, use Kaggle's built-in **Secrets** feature (Add-ons > Secrets) to inject these keys securely instead of hardcoding them in the notebook cells!
