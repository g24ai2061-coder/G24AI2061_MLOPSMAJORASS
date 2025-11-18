DATASET & MODEL DETAILS

DATASET:
• Olivetti Faces Dataset from sklearn
• 400 grayscale images (64x64)
• Flattened into 4096 features

MODEL:
• DecisionTreeClassifier
• StandardScaler normalization
• Saved with joblib as:
{
“model”: DecisionTreeClassifier,
“scaler”: StandardScaler
}
• Saved file: savedmodel.pth

⸻

HOW TO RUN LOCALLY 1. Install Dependencies:
pip install -r requirements.txt 2. Train the Model:
python train.py
This generates:
• savedmodel.pth
• test_data.pkl 3. Evaluate the Model:
python test.py 4. Run the Flask Web App:
python app.py
Open browser:
http://localhost:4000

⸻

DOCKERIZATION

Build Docker image:
docker build -t yourdockerhubname/olivetti-mlops .

Run container:
docker run -p 4000:4000 yourdockerhubname/olivetti-mlops

Push to Docker Hub:
docker push yourdockerhubname/olivetti-mlops

⸻

CI/CD USING GITHUB ACTIONS

GitHub Actions workflow automatically: 1. Checks out code 2. Sets up Python 3. Installs dependencies 4. Runs train.py 5. Runs test.py

Workflow file:
.github/workflows/ci.yml

⸻

KUBERNETES DEPLOYMENT (3 REPLICAS)

Deploy to Kubernetes:
kubectl apply -f k8s-deployment.yaml

Check running pods:
kubectl get pods

Check service:
kubectl get svc

The service exposes the Flask app via LoadBalancer.

⸻

GIT BRANCHING STRATEGY (REQUIRED)

main:
• README.txt
• .gitignore
• Documentation

dev:
• train.py
• test.py
• CI workflow

docker_cicd:
• Dockerfile
• app.py
• k8s-deployment.yaml

⸻

MODEL PERFORMANCE (EXAMPLE)

Training Accuracy: 1.00
Test Accuracy: 0.82
Tree Depth: 27
Leaves: 78

⸻

NOTES
• Only DecisionTreeClassifier is used (as required)
• Model and scaler saved using joblib
• Flask app loads both model and scaler for predictions
• Includes CI/CD, Docker, and Kubernetes
