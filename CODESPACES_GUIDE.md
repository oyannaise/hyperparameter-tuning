# GitHub Codespaces / Docker Playground Deployment Guide

## 📊 Local Machine Results (Baseline)

Training completed successfully on local machine with the following results:

- **Accuracy**: 0.8480 (84.80%)
- **F1 Score**: 0.8946 (89.46%)
- **Validation Loss**: 0.4108
- **Platform**: Local Docker (CPU only)
- **Training Time**: ~15 minutes (3 epochs)

### Hyperparameters Used:
```
Learning Rate: 2.8e-5
Warmup Steps: 150
Scheduler: linear
Batch Size: 16
Weight Decay: 0.01
Epochs: 3
```

---

## 🚀 Option 1: GitHub Codespaces (Recommended)

### Step 1: Push to GitHub

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Add Docker training setup for MRPC"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Launch Codespace

1. Go to your GitHub repository
2. Click the green **"Code"** button
3. Select **"Codespaces"** tab
4. Click **"Create codespace on main"**

### Step 3: Run Training in Codespaces

Once your Codespace is ready:

```bash
# Build and run with Docker Compose
docker-compose up --build
```

Or using Docker directly:

```bash
# Build the image
docker build -t mrpc-training:latest .

# Run training
docker run -v $(pwd)/docker-checkpoints:/app/checkpoints mrpc-training:latest
```

### Step 4: Check Results

After training completes, check the logs:

```bash
# View training output
docker logs $(docker ps -a | grep mrpc-training | awk '{print $1}')
```

---

## 🌐 Option 2: Docker Playground

### Step 1: Access Docker Playground

Go to https://labs.play-with-docker.com/

Click **"Start"** and then **"Add New Instance"**

### Step 2: Upload Your Code

Option A - Clone from GitHub (after pushing):
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

Option B - Manual upload:
1. Create the directory structure
2. Upload files via the web interface or copy-paste content

### Step 3: Run Training

```bash
# Build the image
docker build -t mrpc-training:latest .

# Run training
docker run -v $(pwd)/docker-checkpoints:/app/checkpoints mrpc-training:latest
```

---

## 📝 Performance Comparison Checklist

After running on GitHub Codespaces or Docker Playground, compare:

| Metric | Local Machine | Codespaces | Docker Playground |
|--------|--------------|------------|-------------------|
| **Accuracy** | 0.8480 | ? | ? |
| **F1 Score** | 0.8946 | ? | ? |
| **Val Loss** | 0.4108 | ? | ? |
| **Training Time** | ~15 min | ? | ? |
| **CPU Type** | Apple Silicon/Intel | ? | ? |

---

## 🔍 Expected Differences

### Should be EXACTLY the same:
- ✅ Accuracy (with same random seed=42)
- ✅ F1 Score
- ✅ Validation Loss
- ✅ Model architecture

### Might be different:
- ⚠️ Training time (depends on CPU performance)
- ⚠️ Download speeds (for datasets/models)

---

## 🐛 Potential Issues & Adaptations

### Issue 1: Memory Limits
If you hit memory issues:
```bash
# Reduce batch size in run_training.sh
--batch_size 8  # instead of 16
```

### Issue 2: W&B Offline Mode
By default, W&B is in offline mode. To enable online logging:

1. Get your API key from https://wandb.ai/authorize
2. In `docker-compose.yml`, uncomment and set:
```yaml
environment:
  - WANDB_API_KEY=your_api_key_here
```

### Issue 3: Docker Socket Permission Issues
If you get permission errors:
```bash
sudo docker-compose up --build
```

---

## ✅ Verification Steps

1. **Check that training completes all 3 epochs**
   - Look for "Training Completed!" message
   - Verify exit code is 0

2. **Compare final metrics**
   - Should match local results (±0.001 due to floating point)
   
3. **Check W&B logs** (if using online mode)
   - Go to https://wandb.ai/YOUR_USERNAME/mrpc-docker-training
   - Compare runs visually

4. **Verify checkpoints are saved**
   ```bash
   ls -lh docker-checkpoints/
   ```

---

## 📦 Files Needed for Deployment

Make sure these files are in your repository:

- ✅ `Dockerfile`
- ✅ `docker-compose.yml`
- ✅ `requirements.txt`
- ✅ `run_training.sh`
- ✅ `main.py`
- ✅ `src/` directory (all Python files)
- ✅ `.dockerignore`
- ✅ `.devcontainer/devcontainer.json` (for Codespaces)
- ✅ This README file

---

## 🎯 Success Criteria

Your Task 3 is complete when:

1. ✅ Docker image runs without modifications on Codespaces/Playground
2. ✅ Training completes successfully (3 epochs)
3. ✅ Performance metrics match local results (within ±0.1%)
4. ✅ You've documented any adaptations needed
5. ✅ You've compared and documented the results

---

## 📸 Screenshot Documentation

For your report, capture:

1. GitHub Codespaces terminal showing training completion
2. Final metrics output
3. W&B dashboard comparison (if using online mode)
4. Performance comparison table

