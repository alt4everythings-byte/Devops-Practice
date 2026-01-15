# Lab 4: Kubernetes Deployment

## Steps
1. kubectl apply -f deployment.yaml
2. kubectl get pods
3. kubectl expose deployment ml-deploy --type=NodePort --port=5000
4. kubectl scale deployment ml-deploy --replicas=3
