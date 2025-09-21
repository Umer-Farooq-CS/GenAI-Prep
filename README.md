# Generative AI Practice Exams - Reorganized

---

## 📝 MULTIPLE CHOICE QUESTIONS (All Exams Combined)

### Exam 1 MCQs

1. **Which activation function is most suitable for the output layer in binary classification?**
   a) ReLU
   **b) Sigmoid** ✓
   c) Tanh
   d) Linear

2. **What is the main advantage of using dilated convolutions?**
   a) Reduces computational cost
   **b) Increases receptive field without increasing parameters** ✓
   c) Improves gradient flow
   d) Prevents overfitting

3. **In a VAE, what does the reparameterization trick enable?**
   a) Faster training
   b) Better reconstruction
   **c) Gradient flow through random sampling** ✓
   d) Reduced overfitting

4. **What problem does batch normalization primarily solve?**
   a) Overfitting
   b) Vanishing gradients
   c) Mode collapse
   **d) Internal covariate shift** ✓

5. **In GANs, what indicates successful training?**
   a) Generator loss approaching zero
   **b) Discriminator accuracy around 50%** ✓
   c) Discriminator loss approaching zero
   d) Perfect reconstruction

6. **What is the key difference between causal and standard convolutions?**
   a) Kernel size
   b) Padding type
   **c) Temporal dependencies** ✓
   d) Number of channels

7. **Which technique helps prevent mode collapse in GANs?**
   a) Higher learning rate
   **b) Mini-batch discrimination** ✓
   c) More epochs
   d) Larger batch size

8. **What does the receptive field determine in CNNs?**
   a) Output size
   b) Number of parameters
   **c) Input region affecting output** ✓
   d) Training speed

9. **In autoregressive models, what does the chain rule help compute?**
   a) Gradient descent
   **b) Joint probability distribution** ✓
   c) Loss function
   d) Activation functions

10. **What is the main limitation of finite memory models?**
    a) High computational cost
    **b) Cannot capture long-range dependencies** ✓
    c) Require large datasets
    d) Poor accuracy

11. **Which loss function is commonly used for multi-class classification?**
    a) MSE
    b) Binary cross-entropy
    **c) Categorical cross-entropy** ✓
    d) L1 loss

12. **What happens when learning rate is too high?**
    a) Slow convergence
    b) Underfitting
    **c) Overshooting minimum** ✓
    d) Perfect training

13. **In PixelRNN, what determines the order of pixel generation?**
    a) Random sampling
    **b) Spatial ordering (top-left to bottom-right)** ✓
    c) Color intensity
    d) Feature importance

14. **What is weight sharing in CNNs?**
    a) Using same weights across different layers
    **b) Using same filter weights across spatial locations** ✓
    c) Sharing weights between networks
    d) Copying weights from pre-trained models

15. **Which component is NOT part of LSTM?**
    a) Forget gate
    b) Input gate
    **c) Memory gate** ✓
    d) Output gate

16. **What does ELBO stand for in VAEs?**
    a) Expected Lower Bound Optimization
    **b) Evidence Lower Bound** ✓
    c) Entropy Lower Bound Objective
    d) Error Lower Bound Operation

17. **In DCGANs, which activation is used in the generator's output layer?**
    a) ReLU
    b) Sigmoid
    **c) Tanh** ✓
    d) LeakyReLU

18. **What is the purpose of masked convolutions in PixelCNN?**
    a) Reduce overfitting
    **b) Prevent future information leakage** ✓
    c) Improve accuracy
    d) Speed up training

19. **Which optimizer is commonly recommended for GAN training?**
    a) SGD
    b) RMSprop
    **c) Adam** ✓
    d) Adagrad

20. **What does transfer learning primarily help with?**
    a) Faster inference
    **b) Limited data scenarios** ✓
    c) Larger models
    d) Better accuracy

### Exam 2 MCQs

21. **What is the main advantage of using LeakyReLU over ReLU?**
    a) Faster computation
    **b) Prevents dying neurons** ✓
    c) Better accuracy
    d) Smoother gradients

22. **In GANs, what does the discriminator try to minimize?**
    a) Generator loss
    **b) Real vs fake classification error** ✓
    c) Reconstruction error
    d) KL divergence

23. **What is the key innovation of ResNet architecture?**
    a) Batch normalization
    **b) Skip connections** ✓
    c) Depthwise convolutions
    d) Attention mechanism

24. **Which technique is used to handle the posterior collapse problem in VAEs?**
    a) Higher learning rate
    **b) β-VAE (beta weighting)** ✓
    c) More epochs
    d) Larger latent dimension

25. **What does stride=2 in convolution achieve?**
    a) Increases receptive field
    **b) Reduces spatial dimensions by half** ✓
    c) Doubles the number of features
    d) Improves accuracy

26. **In RNNs, what causes the exploding gradient problem?**
    a) Small weight values
    **b) Large weight values being multiplied repeatedly** ✓
    c) Too many layers
    d) Wrong activation function

27. **What is the purpose of the forget gate in LSTM?**
    **a) Remove irrelevant information from cell state** ✓
    b) Add new information
    c) Generate output
    d) Control input flow

28. **Which loss function is appropriate for regression tasks?**
    a) Cross-entropy
    **b) Mean Squared Error** ✓
    c) Hinge loss
    d) Focal loss

29. **What does global average pooling do?**
    a) Finds maximum value
    **b) Computes average across spatial dimensions** ✓
    c) Reduces number of channels
    d) Increases resolution

30. **In PixelRNN, how are color channels processed?**
    a) Simultaneously
    **b) Sequentially (R→G→B)** ✓
    c) Randomly
    d) In parallel

31. **What is the main benefit of using pre-trained models?**
    a) Faster training
    **b) Better generalization with limited data** ✓
    c) Smaller model size
    d) Lower computational cost

32. **Which normalization technique is applied along the batch dimension?**
    a) Layer normalization
    **b) Batch normalization** ✓
    c) Group normalization
    d) Instance normalization

33. **What problem does dropout help solve?**
    a) Underfitting
    **b) Overfitting** ✓
    c) Vanishing gradients
    d) Mode collapse

34. **In autoencoders, what is the bottleneck layer?**
    a) Input layer
    b) Output layer
    **c) Latent representation** ✓
    d) Hidden layer

35. **What is the purpose of padding in convolutions?**
    a) Increase computational efficiency
    **b) Preserve spatial dimensions** ✓
    c) Reduce overfitting
    d) Improve accuracy

36. **Which activation function has the range [-1, 1]?**
    a) Sigmoid
    b) ReLU
    **c) Tanh** ✓
    d) Softmax

37. **What does temperature parameter control in softmax?**
    a) Learning rate
    **b) Distribution sharpness** ✓
    c) Model complexity
    d) Training speed

38. **In DCGANs, what replaces pooling layers?**
    a) Dropout
    **b) Strided convolutions** ✓
    c) Skip connections
    d) Attention

39. **What is the main challenge in training GANs?**
    a) High computational cost
    **b) Training instability** ✓
    c) Large datasets required
    d) Complex architecture

40. **What does the encoder in VAE approximate?**
    a) Prior distribution
    **b) Posterior distribution** ✓
    c) Likelihood function
    d) Evidence

### Exam 3 MCQs

41. **What is the main difference between autoencoders and VAEs?**
    a) Number of layers
    **b) Probabilistic vs deterministic encoding** ✓
    c) Loss function
    d) Training speed

42. **In CNN architectures, what does 1×1 convolution primarily do?**
    a) Increase spatial resolution
    **b) Change number of channels** ✓
    c) Add non-linearity
    d) Reduce overfitting

43. **What is the role of the KL divergence term in VAE loss?**
    a) Reconstruction quality
    **b) Regularization of latent space** ✓
    c) Training stability
    d) Gradient flow

44. **Which component is essential for causal modeling in sequence generation?**
    a) Attention mechanism
    **b) Masked convolutions** ✓
    c) Batch normalization
    d) Skip connections

45. **What does the receptive field size determine in CNNs?**
    a) Model accuracy
    b) Training time
    **c) Context window for each output** ✓
    d) Number of parameters

46. **In RNN training, what technique helps with exploding gradients?**
    **a) Gradient clipping** ✓
    b) Batch normalization
    c) Dropout
    d) Skip connections

47. **What is the primary goal of the generator in GANs?**
    a) Classify real vs fake data
    **b) Fool the discriminator** ✓
    c) Minimize reconstruction error
    d) Learn feature representations

48. **Which pooling operation retains the most information?**
    a) Max pooling
    b) Average pooling
    c) Min pooling
    **d) No pooling** ✓

49. **What does the softmax function ensure?**
    a) Non-negative outputs
    **b) Sum of outputs equals 1** ✓
    c) Differentiable function
    d) Faster computation

50. **In PixelCNN, what advantage does it have over PixelRNN?**
    a) Better quality
    **b) Parallel training** ✓
    c) Smaller model size
    d) Higher accuracy

51. **What is transfer learning most useful for?**
    a) Large datasets
    **b) Small datasets** ✓
    c) Fast inference
    d) Model compression

52. **Which activation function can cause vanishing gradient problem?**
    a) ReLU
    **b) Sigmoid** ✓
    c) LeakyReLU
    d) ELU

53. **What does early stopping prevent?**
    a) Underfitting
    **b) Overfitting** ✓
    c) Slow convergence
    d) Memory overflow

54. **In denoising autoencoders, what is added to the input?**
    a) Labels
    **b) Noise** ✓
    c) Features
    d) Regularization

55. **What is the main challenge with very deep networks?**
    a) High computational cost
    **b) Vanishing/exploding gradients** ✓
    c) Overfitting
    d) Memory requirements

56. **Which loss function is used for binary classification?**
    a) MSE
    **b) Binary cross-entropy** ✓
    c) Categorical cross-entropy
    d) L1 loss

57. **What does batch size affect in training?**
    a) Model accuracy only
    **b) Training stability and speed** ✓
    c) Model size
    d) Inference time

58. **In GANs, what indicates mode collapse?**
    a) High discriminator accuracy
    **b) Low generator diversity** ✓
    c) Fast convergence
    d) Perfect reconstruction

59. **What is the purpose of residual connections?**
    a) Reduce parameters
    **b) Enable training of deeper networks** ✓
    c) Improve accuracy
    d) Prevent overfitting

60. **Which technique helps with internal covariate shift?**
    a) Dropout
    **b) Batch normalization** ✓
    c) Weight decay
    d) Data augmentation

---

## 📋 SHORT QUESTIONS (All Exams Combined)

### Exam 1 Short Questions

1. **Explain the difference between generative and discriminative models. Give one example of each.**
   
   **Answer:** Generative models learn the joint probability distribution P(X,Y) and can generate new data samples. Discriminative models learn the conditional probability P(Y|X) for classification. Examples: GAN (generative), CNN classifier (discriminative).

2. **What are the three main components of a VAE architecture? Briefly explain each.**
   
   **Answer:** 
   - **Encoder:** Maps input to latent space parameters (mean μ and variance σ)
   - **Latent space:** Probabilistic representation where sampling occurs
   - **Decoder:** Reconstructs data from latent code

3. **Why do we use the reparameterization trick in VAEs? What problem does it solve?**
   
   **Answer:** Enables gradient flow through random sampling by making sampling differentiable. Formula: z = μ + σε where ε ~ N(0,1). Solves the problem of non-differentiable random sampling.

4. **Describe the concept of receptive field in CNNs. How can we increase it without adding more parameters?**
   
   **Answer:** Receptive field is the region of input that affects each output neuron. Can be increased using dilated convolutions - expanding spacing between kernel elements to cover larger areas.

5. **What is the vanishing gradient problem? How does it affect deep neural network training?**
   
   **Answer:** Gradients become very small during backpropagation, causing slow weight updates and poor learning in deep networks. This prevents effective training of deep layers.

6. **Explain the working principle of masked convolutions in autoregressive models.**
   
   **Answer:** Zero out future positions in convolution kernels to prevent the model from seeing future information during training, ensuring causal generation.

### Exam 2 Short Questions

7. **Explain the concept of weight sharing in CNNs. What are its advantages?**
   
   **Answer:** Using the same filter weights across all spatial locations in the input. Advantages: reduces parameters, enables translation invariance, improves generalization.

8. **What is mode collapse in GANs? Describe one technique to address it.**
   
   **Answer:** Generator produces limited variety of outputs, focusing on a few modes. Technique: Mini-batch discrimination - discriminator sees multiple samples simultaneously to encourage diversity.

9. **Compare Row LSTM and Diagonal BiLSTM in terms of their processing patterns.**
   
   **Answer:** Row LSTM processes pixels row by row sequentially. Diagonal BiLSTM processes along diagonal directions with bidirectional information flow, capturing more spatial dependencies.

10. **What is the difference between parameters and hyperparameters? Give examples of each.**
    
    **Answer:** Parameters are learned during training (weights, biases). Hyperparameters are set before training (learning rate, batch size, architecture choices).

11. **Explain how dilated convolutions increase receptive field. Draw a simple diagram.**
    
    **Answer:** Dilated convolutions insert gaps between kernel elements, allowing the same kernel to cover larger input areas without increasing parameters.

12. **What is the curse of dimensionality in the context of probabilistic modeling?**
    
    **Answer:** As dimensions increase, data becomes sparse, making probability estimation difficult and requiring exponentially more data for reliable estimates.

### Exam 3 Short Questions

13. **Explain the reparameterization trick in VAEs with the mathematical formulation.**
    
    **Answer:** z = μ + σε where ε ~ N(0,1). This makes sampling differentiable by separating the stochastic part (ε) from the deterministic parameters (μ, σ).

14. **What are the key architectural guidelines for DCGANs? List at least four.**
    
    **Answer:** 
    - Replace pooling with strided convolutions
    - Use batch normalization in generator and discriminator
    - Remove fully connected hidden layers
    - Use ReLU in generator, LeakyReLU in discriminator
    - Use Tanh activation in generator output

15. **Compare the advantages and disadvantages of RNN vs CNN for sequential data.**
    
    **Answer:** RNN: handles variable length, captures temporal dependencies, but slow training. CNN: parallel training, faster, but limited temporal modeling.

16. **Explain how mini-batch discrimination helps prevent mode collapse in GANs.**
    
    **Answer:** Discriminator sees multiple samples simultaneously, computing statistics across the batch to encourage generator diversity and prevent mode collapse.

17. **What is the difference between Mask A and Mask B in PixelCNN? When is each used?**
    
    **Answer:** Mask A: blocks center pixel (used for first layer). Mask B: allows center pixel (used for subsequent layers). Ensures causal generation.

18. **Describe the concept of latent variable models and their applications.**
    
    **Answer:** Models that use unobserved variables to explain observed data. Applications: VAEs for generation, factor analysis, probabilistic PCA.

---

## 🧮 LONG & NUMERICAL QUESTIONS (All Exams Combined)

### Exam 1 Long Questions

1. **Neural Network Computation (15 marks)**
   
   Consider a simple neural network with:
   - 2 inputs: x₁ = 0.8, x₂ = 0.3
   - 1 hidden neuron with sigmoid activation
   - 1 output neuron with linear activation
   - Weights: w₁ = 0.5, w₂ = -0.3, w₃ = 0.7
   - Biases: b₁ = 0.2, b₂ = -0.1
   - Target output: y = 0.9
   - Learning rate: η = 0.2
   
   a) Perform forward propagation and calculate the output (5 marks)
   b) Calculate the loss using MSE (3 marks)
   c) Perform backpropagation and update all weights and biases (7 marks)

   **Final Answer:** 
   - Output: 0.65
   - MSE Loss: 0.0625
   - Updated weights: w₁ = 0.52, w₂ = -0.28, w₃ = 0.72, b₁ = 0.22, b₂ = -0.08

2. **CNN Parameter Calculation (15 marks)**
   
   A CNN architecture has the following layers:
   - Input: 64×64×3 image
   - Conv1: 32 filters of size 5×5, stride=1, padding=2
   - MaxPool1: 2×2, stride=2
   - Conv2: 64 filters of size 3×3, stride=1, padding=1
   - MaxPool2: 2×2, stride=2
   - Flatten and FC layer with 128 neurons
   - Output layer with 10 neurons (classification)
   
   Calculate:
   a) Output dimensions after each layer (8 marks)
   b) Total number of parameters in the network (7 marks)

   **Final Answer:**
   - Output dimensions: 64×64×32 → 32×32×32 → 32×32×64 → 16×16×64 → 16384 → 10
   - Total parameters: 2,425,610

### Exam 2 Long Questions

3. **Autoregressive Model Design (15 marks)**
   
   For the sentence "AI is transforming technology", design a finite memory model with window size 2.
   
   a) Write down all the conditional probability equations needed to model this sentence (8 marks)
   b) Explain the main limitation of this approach (3 marks)
   c) Suggest how RNN-based models overcome this limitation (4 marks)

   **Final Answer:**
   - P(AI|START, START), P(is|START, AI), P(transforming|AI, is), P(technology|is, transforming)
   - Limitation: Cannot capture long-range dependencies beyond window size
   - RNNs use hidden states to maintain information across arbitrary distances

4. **GAN Training Analysis (15 marks)**
   
   You are training a GAN with the following observations:
   - Epoch 1-10: Discriminator accuracy = 95%, Generator loss = 5.2
   - Epoch 11-20: Discriminator accuracy = 52%, Generator loss = 0.8
   - Epoch 21-30: Discriminator accuracy = 50%, Generator loss = 0.7
   
   a) Analyze what's happening in each phase of training (6 marks)
   b) Identify which phase shows the best training progress and why (4 marks)
   c) Suggest two techniques to improve training stability (5 marks)

   **Final Answer:**
   - Phase 1: Discriminator dominates, generator learning
   - Phase 2: Balanced competition, optimal training
   - Phase 3: Near equilibrium, potential mode collapse
   - Best phase: 11-20 (balanced competition)
   - Techniques: Learning rate scheduling, gradient penalty

### Exam 3 Long Questions

5. **VAE Implementation Analysis (15 marks)**
   
   A VAE is trained on MNIST with the following specifications:
   - Encoder: 784 → 512 → 256 → 20 (mean and log_var outputs)
   - Latent dimension: 10
   - Decoder: 10 → 256 → 512 → 784
   - β = 1.0 (standard VAE)
   
   a) Calculate the KL divergence for latent code z = [0.5, -0.2] with μ = [0.3, -0.1] and σ = [0.8, 0.6] (6 marks)
   b) If generated images are blurry, what could be the problem and how would you fix it? (5 marks)
   c) How would you modify the loss function to improve disentanglement? (4 marks)

   **Final Answer:**
   - KL divergence: 0.45
   - Blurry images: Use β-VAE with β > 1, or increase reconstruction weight
   - Disentanglement: Use β-VAE with β > 1, or add total correlation penalty

6. **Convolutional Architecture Design (15 marks)**
   
   Design a CNN for generating 32×32 RGB images using the following constraints:
   - Start with 100-dimensional noise vector
   - Use transpose convolutions for upsampling
   - Include batch normalization
   - Final output should be in range [-1, 1]
   
   a) Design the complete architecture with layer specifications (10 marks)
   b) Calculate the number of parameters in each layer (5 marks)

   **Final Answer:**
   - Architecture: FC(100→1024) → Reshape(4×4×256) → ConvT(256→128) → ConvT(128→64) → ConvT(64→3)
   - Parameters: 102,400 + 524,288 + 131,072 + 12,288 = 770,048 total

---

## 📚 ADDITIONAL PRACTICE QUESTIONS

### Quick MCQ Set (20 questions)

1. **What is the main advantage of using Adam optimizer?**
   a) Faster convergence than SGD
   **b) Adaptive learning rates for each parameter** ✓
   c) Uses less memory
   d) Better for small datasets

2. **In GANs, what is the ideal discriminator accuracy during training?**
   a) 100%
   b) 0%
   **c) 50%** ✓
   d) 75%

3. **What does the β parameter control in β-VAE?**
   a) Learning rate
   **b) Weight of KL divergence term** ✓
   c) Batch size
   d) Latent dimension

4. **Which technique is used to train very deep networks (100+ layers)?**
   a) Dropout
   b) Batch normalization
   **c) Residual connections** ✓
   d) Data augmentation

5. **What is the main limitation of standard RNNs?**
   a) High computational cost
   **b) Cannot handle long sequences** ✓
   c) Require large datasets
   d) Poor accuracy

### Numerical Problems

1. **Convolution Output Size:**
   Input: 28×28, Kernel: 5×5, Stride: 2, Padding: 1
   **Answer:** Output = 14×14

2. **Parameter Count:**
   Dense layer: 512 inputs → 256 outputs
   **Answer:** Parameters = (512 × 256) + 256 = 131,328

3. **Probability Calculation:**
   Given logits [2.1, 1.5, 0.8] for 3 classes, calculate softmax probabilities.
   **Answer:** [0.67, 0.24, 0.09]

---

*This reorganized practice exam covers all key concepts from your syllabus and matches the style of your original quizzes. The questions progress from basic concepts to more complex applications, exactly like what you'd expect in your actual exam.*
