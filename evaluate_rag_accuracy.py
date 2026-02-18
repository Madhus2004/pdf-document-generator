import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Questions
questions = [
    "What is the definition of Machine Learning according to Arthur Samuel?",
    "Define supervised learning.",
    "What is the difference between classification and regression?",
    "List the four types of Machine Learning.",
    "What is reinforcement learning?"
]

# Ground Truth Answers
ground_truth = [
    "The field of study that gives computers the ability to learn without being explicitly programmed.",
    "Supervised learning uses labeled data to train models and map inputs to known outputs.",
    "Classification predicts categorical outputs while regression predicts continuous numerical values.",
    "Supervised, Unsupervised, Semi-Supervised and Reinforcement learning.",
    "Reinforcement learning is learning by rewards to maximize total reward."
]

# Replace this with your RAG responses
rag_answers = [
    "According to Arthur Samuel, 'Machine Learning' is defined as 'the field of study that gives computers the ability to learn without being explicitly programmed.' This definition was coined by him in 1959 while he was at IBM.",
    "Supervised learning is a subcategory of machine learning that focuses on learning a classification or regression model from labeled training data, which consists of inputs along with the desired outputs or targets. The goal is to create an accurate predictive model for various real-world problems such as fraud detection and spam filtering. In supervised learning, there is an exact idea about the classes of objects, and it requires a significant amount of computation time during training. Some popular algorithms used in supervised learning include linear regression, logistic regression, support vector machine, K nearest neighbor, decision tree, random forest, naive Bayes, and neural networks.",
    "Classification and regression are two fundamental types of supervised machine learning techniques used for predictive analysis. The key difference between them lies in the nature of output variables they deal with: 1. **Regression**: - Regression is a technique used when the output variable can take continuous numerical values, such as price of a car, delivery time, or credit limit. - The goal is to model the relationship between independent variables (features) and a dependent variable (target). - Popular regression algorithms include Linear Regression, Regression Trees, Non-Linear Regression, Bayesian Linear Regression, and Polynomial Regression. 2. **Classification**: - Classification is used when the output variable takes categorical or discrete (non-continuous) values, such as whether an email is spam, or whether a transaction is fraudulent. - The goal is to classify instances based on their features and assign them to specific categories or classes. - Popular classification algorithms include Random Forest, Decision Trees, Logistic Regression, and Support Vector Machines.",
    "The four types of Machine Learning are: 1. Supervised Machine Learning 2. Unsupervised Machine Learning 3. Semi-supervised learning 4. Reinforcement Learning",
    "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent takes actions to maximize cumulative rewards over time. Unlike supervised learning, RL does not require labeled input/output pairs; instead, it learns from the consequences of its actions via rewards and penalties. In reinforcement learning, the agent gets rewarded for each good action and punished for each bad action, and the goal is to maximize the rewards. The process works on a feedback-based system where the AI agent automatically explores its surroundings, takes actions, learns from experiences, and improves its performance. The reinforcement learning process is similar to how a human being learns new things through experiences in their daily life. An example of reinforcement learning is playing a game, where the Game is the environment, moves of an agent at each step define states, and the goal of the agent is to get a high score. A reinforcement learning problem can be formalized using Markov Decision Process (MDP), in which the agent constantly interacts with the environment and performs actions, at each action, the environment responds and generates a new state. The key components of RL are the Agent, State (S), Policy (π), Action (A), Reward (R), and Environment. The advantages of reinforcement learning include solving complex real-world problems that are difficult to be solved by general techniques and being somewhat related to unsupervised learning but constituting its own category of machine learning. Typical applications of reinforcement learning involve playing games, such as chess, Go, Atari video games, and some form of robots, like drones, warehouse robots, and more recently self-driving cars."
]

def evaluate_accuracy(gt, pred):
    gt_embeddings = model.encode(gt)
    pred_embeddings = model.encode(pred)

    similarities = []
    for i in range(len(gt)):
        sim = cosine_similarity(
            [gt_embeddings[i]],
            [pred_embeddings[i]]
        )[0][0]
        similarities.append(sim)

    return similarities, np.mean(similarities)

scores, avg_score = evaluate_accuracy(ground_truth, rag_answers)

print("\n===== RAG Accuracy Evaluation =====")
for i, score in enumerate(scores):
    print(f"Q{i+1} Similarity Score: {score:.4f}")

print(f"\nOverall Semantic Accuracy: {avg_score:.4f}")
