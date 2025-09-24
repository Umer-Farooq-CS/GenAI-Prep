# Deep Learning for Perception Course Topics - Hierarchical Structure

## I. Machine Learning Fundamentals

### A. Introduction to Learning
- Definition of learning vs. traditional programming
- Examples of complex problems (image recognition, face detection, tumor classification)
- Arthur Samuel's definition of machine learning (1959)
- Tom Mitchell's formal definition (Task T, Performance P, Experience E)

### B. Types of Machine Learning
1. **Supervised Learning**
   - Definition and characteristics
   - Classification vs. Regression problems
   - Examples: spam detection, medical imaging, language translation
   - Common algorithms: Linear Regression, Logistic Regression, SVMs, Decision Trees, Neural Networks, k-NN

2. **Unsupervised Learning**
   - Definition and characteristics
   - Applications: clustering, dimensionality reduction, data visualization, anomaly detection
   - Common algorithms: K-Means, PCA, Hierarchical Cluster Analysis, One-class SVM

3. **Reinforcement Learning**
   - Agent-environment interaction
   - Rewards and policies
   - Examples: game playing, robotics, AlphaGo

## II. Machine Learning Workflow and Methodology

### A. ML Project Workflow
1. Data Gathering and Preparation
2. Feature Extraction/Selection
3. Model Selection and Development
4. Training and Testing
5. Model Deployment
6. Model Monitoring and Management

### B. Data Management
- Data preprocessing and normalization
- Handling noisy and missing data
- Data quality importance
- Training/Validation/Test set splits

### C. Model Evaluation and Validation
1. **Cross-Validation Techniques**
   - K-Fold Cross Validation
   - Leave-One-Out Cross Validation (LOOCV)
   - When to use each method

2. **Bias-Variance Tradeoff**
   - Understanding bias (underfitting)
   - Understanding variance (overfitting)
   - Mitigation strategies for each

3. **Hyperparameters vs. Parameters**
   - Definitions and differences
   - Parameter learning during training
   - Hyperparameter tuning methods

## III. Linear Models and Optimization

### A. Linear Models
- Linear separability assumption
- Decision boundaries and hyperplanes
- Linear regression fundamentals
- Least squares fitting
- Probabilistic linear regression

### B. Loss Functions
1. **Mean Squared Error (MSE)**
   - Mathematical formulation
   - When to use MSE
   - Advantages: convexity, differentiability, interpretability
   - Limitations: sensitivity to outliers
   - Use cases: engineering, financial modeling, medical applications

### C. Optimization Algorithms
1. **Gradient Descent**
   - Basic principles and mathematical foundation
   - Parameter update rules
   - Learning rate considerations
   - Convergence properties

2. **Hyperparameter Tuning**
   - Grid search vs. random search
   - Comparison of search strategies

## IV. Neural Networks Fundamentals

### A. Biological Inspiration
- Brain vs. machine architecture comparison
- Biological neuron structure (dendrites, cell body, axon, synapses)
- Neural signal propagation and thresholds
- Massive parallelism in biological systems

### B. Artificial Neural Networks Basics
- Basic neuron model (perceptron)
- Weighted inputs and activation functions
- Network architectures and connections
- Applications: ALVINN autonomous driving system

### C. Perceptrons
1. **Rosenblatt's Perceptron (1957)**
   - Single-layer model
   - Mathematical formulation
   - Decision boundaries
   - Linear separability limitations

2. **Perceptron Learning Algorithm**
   - Weight update rules
   - Learning rate parameter
   - Training procedures
   - Convergence properties

3. **Representational Limitations**
   - Linearly separable vs. non-linearly separable problems
   - XOR problem as classic example
   - Need for multi-layer networks

## V. Advanced Neural Networks

### A. Activation Functions
1. **Sigmoid Function**
   - Mathematical properties
   - Output range [0,1]
   - Gradient saturation problems
   - Vanishing gradient issues

2. **Tanh Function**
   - Output range [-1,1]
   - Zero-centered outputs
   - Still suffers from saturation

3. **ReLU (Rectified Linear Unit)**
   - Mathematical simplicity: f(x) = max(0,x)
   - Advantages: faster training, computational efficiency, sparsity
   - Solving vanishing gradient problem
   - Dying ReLU problem

### B. Multi-layer Networks
- Overcoming linear separability limitations
- Hidden layers and their role
- Non-linear decision boundaries
- Backpropagation algorithm introduction

### C. Training Algorithms
1. **Gradient Descent Variants**
   - Standard (batch) gradient descent
   - Stochastic gradient descent
   - Training procedures and weight updates

2. **Backpropagation**
   - Algorithm for multi-layer networks
   - Forward and backward passes
   - Chain rule application
   - Local minima challenges

### D. Training Considerations
- Momentum in gradient descent
- Avoiding local minima
- Multiple random initializations
- Convergence criteria and stopping conditions

## VI. Practical Considerations

### A. Problem Suitability for Neural Networks
- Noisy training data tolerance
- Complex sensor data processing
- High-dimensional input spaces
- Real-valued and discrete outputs
- Long training vs. fast evaluation tradeoffs

### B. Network Design Decisions
- Architecture selection
- Layer sizes and depths
- Activation function choices
- Initialization strategies
- Regularization needs

### C. Performance Evaluation
- Training vs. validation vs. test performance
- Overfitting detection and prevention
- Model generalization assessment
- Cross-validation in neural network context