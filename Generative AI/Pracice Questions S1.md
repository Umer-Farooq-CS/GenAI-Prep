# Generative AI Practice Exams - Reorganized

---

## 📝 MULTIPLE CHOICE QUESTIONS (All Exams Combined)

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

### 1. Neural Network Computation (15 marks)

**Problem Statement:**
Consider a simple neural network with:
- 2 inputs: x₁ = 0.8, x₂ = 0.3
- 1 hidden neuron with sigmoid activation
- 1 output neuron with linear activation
- Weights: w₁ = 0.5, w₂ = -0.3, w₃ = 0.7
- Biases: b₁ = 0.2, b₂ = -0.1
- Target output: y = 0.9
- Learning rate: η = 0.2

**a) Perform forward propagation and calculate the output (5 marks)**

**Step-by-Step Solution:**

**Step 1: Calculate hidden layer input**
```
h_input = w₁ × x₁ + w₂ × x₂ + b₁
h_input = 0.5 × 0.8 + (-0.3) × 0.3 + 0.2
h_input = 0.4 - 0.09 + 0.2
h_input = 0.51
```

**Step 2: Apply sigmoid activation to hidden layer**
```
h_output = σ(h_input) = 1/(1 + e^(-0.51))
h_output = 1/(1 + e^(-0.51))
h_output = 1/(1 + 0.6005)
h_output = 1/1.6005
h_output = 0.625
```

**Step 3: Calculate output layer input**
```
o_input = w₃ × h_output + b₂
o_input = 0.7 × 0.625 + (-0.1)
o_input = 0.4375 - 0.1
o_input = 0.3375
```

**Step 4: Apply linear activation (no change)**
```
output = o_input = 0.3375
```

**Final Answer: Output = 0.3375**

**b) Calculate the loss using MSE (3 marks)**

**Step-by-Step Solution:**

**Step 1: Apply MSE formula**
```
MSE = (1/2) × (target - output)²
MSE = (1/2) × (0.9 - 0.3375)²
MSE = (1/2) × (0.5625)²
MSE = (1/2) × 0.3164
MSE = 0.1582
```

**Final Answer: MSE Loss = 0.1582**

**c) Perform backpropagation and update all weights and biases (7 marks)**

**Step-by-Step Solution:**

**Step 1: Calculate output layer gradient**
```
∂L/∂o_input = output - target = 0.3375 - 0.9 = -0.5625
```

**Step 2: Calculate gradients for w₃ and b₂**
```
∂L/∂w₃ = ∂L/∂o_input × h_output = -0.5625 × 0.625 = -0.3516
∂L/∂b₂ = ∂L/∂o_input = -0.5625
```

**Step 3: Calculate hidden layer gradient**
```
∂L/∂h_output = ∂L/∂o_input × w₃ = -0.5625 × 0.7 = -0.3938
∂L/∂h_input = ∂L/∂h_output × σ'(h_input)
σ'(h_input) = σ(h_input) × (1 - σ(h_input)) = 0.625 × (1 - 0.625) = 0.2344
∂L/∂h_input = -0.3938 × 0.2344 = -0.0923
```

**Step 4: Calculate gradients for w₁, w₂, and b₁**
```
∂L/∂w₁ = ∂L/∂h_input × x₁ = -0.0923 × 0.8 = -0.0738
∂L/∂w₂ = ∂L/∂h_input × x₂ = -0.0923 × 0.3 = -0.0277
∂L/∂b₁ = ∂L/∂h_input = -0.0923
```

**Step 5: Update weights and biases**
```
w₁_new = w₁ - η × ∂L/∂w₁ = 0.5 - 0.2 × (-0.0738) = 0.5148
w₂_new = w₂ - η × ∂L/∂w₂ = -0.3 - 0.2 × (-0.0277) = -0.2945
w₃_new = w₃ - η × ∂L/∂w₃ = 0.7 - 0.2 × (-0.3516) = 0.7703
b₁_new = b₁ - η × ∂L/∂b₁ = 0.2 - 0.2 × (-0.0923) = 0.2185
b₂_new = b₂ - η × ∂L/∂b₂ = -0.1 - 0.2 × (-0.5625) = 0.0125
```

**Final Answer:**
- Updated weights: w₁ = 0.5148, w₂ = -0.2945, w₃ = 0.7703
- Updated biases: b₁ = 0.2185, b₂ = 0.0125

### 2. CNN Parameter Calculation (15 marks)

**Problem Statement:**
A CNN architecture has the following layers:
- Input: 64×64×3 image
- Conv1: 32 filters of size 5×5, stride=1, padding=2
- MaxPool1: 2×2, stride=2
- Conv2: 64 filters of size 3×3, stride=1, padding=1
- MaxPool2: 2×2, stride=2
- Flatten and FC layer with 128 neurons
- Output layer with 10 neurons (classification)

**a) Output dimensions after each layer (8 marks)**

**Step-by-Step Solution:**

**Step 1: Input layer**
```
Input: 64 × 64 × 3
```

**Step 2: Conv1 layer**
```
Formula: output_size = (input_size + 2×padding - kernel_size) / stride + 1
output_size = (64 + 2×2 - 5) / 1 + 1 = (64 + 4 - 5) + 1 = 64
Output: 64 × 64 × 32
```

**Step 3: MaxPool1 layer**
```
Formula: output_size = (input_size - kernel_size) / stride + 1
output_size = (64 - 2) / 2 + 1 = 62/2 + 1 = 32
Output: 32 × 32 × 32
```

**Step 4: Conv2 layer**
```
output_size = (32 + 2×1 - 3) / 1 + 1 = (32 + 2 - 3) + 1 = 32
Output: 32 × 32 × 64
```

**Step 5: MaxPool2 layer**
```
output_size = (32 - 2) / 2 + 1 = 30/2 + 1 = 16
Output: 16 × 16 × 64
```

**Step 6: Flatten layer**
```
output_size = 16 × 16 × 64 = 16,384
Output: 16,384
```

**Step 7: FC layer**
```
Output: 128
```

**Step 8: Output layer**
```
Output: 10
```

**Final Answer:**
- Input: 64×64×3
- Conv1: 64×64×32
- MaxPool1: 32×32×32
- Conv2: 32×32×64
- MaxPool2: 16×16×64
- Flatten: 16,384
- FC: 128
- Output: 10

**b) Total number of parameters in the network (7 marks)**

**Step-by-Step Solution:**

**Step 1: Conv1 parameters**
```
Each filter: 5×5×3 = 75 weights + 1 bias = 76 parameters
Total filters: 32
Conv1 parameters = 32 × 76 = 2,432
```

**Step 2: Conv2 parameters**
```
Each filter: 3×3×32 = 288 weights + 1 bias = 289 parameters
Total filters: 64
Conv2 parameters = 64 × 289 = 18,496
```

**Step 3: FC layer parameters**
```
Weights: 16,384 × 128 = 2,097,152
Bias: 128
FC parameters = 2,097,152 + 128 = 2,097,280
```

**Step 4: Output layer parameters**
```
Weights: 128 × 10 = 1,280
Bias: 10
Output parameters = 1,280 + 10 = 1,290
```

**Step 5: Total parameters**
```
Total = 2,432 + 18,496 + 2,097,280 + 1,290 = 2,119,498
```

**Final Answer: Total parameters = 2,119,498**

### 3. Autoregressive Model Design (15 marks)

**Problem Statement:**
For the sentence "AI is transforming technology", design a finite memory model with window size 2.

**a) Write down all the conditional probability equations needed to model this sentence (8 marks)**

**Step-by-Step Solution:**

**Step 1: Identify the sequence**
```
Sequence: [START, START, "AI", "is", "transforming", "technology"]
```

**Step 2: Apply finite memory with window size 2**
```
For each position, we look at the previous 2 words to predict the next word.
```

**Step 3: Write conditional probability equations**
```
P("AI" | START, START) = probability of "AI" given two START tokens
P("is" | START, "AI") = probability of "is" given START and "AI"
P("transforming" | "AI", "is") = probability of "transforming" given "AI" and "is"
P("technology" | "is", "transforming") = probability of "technology" given "is" and "transforming"
```

**Final Answer:**
- P("AI" | START, START)
- P("is" | START, "AI")
- P("transforming" | "AI", "is")
- P("technology" | "is", "transforming")

**b) Explain the main limitation of this approach (3 marks)**

**Step-by-Step Solution:**

**Step 1: Identify the core limitation**
```
The finite memory model can only look back a fixed number of steps (window size 2).
```

**Step 2: Explain the consequences**
```
- Cannot capture long-range dependencies beyond the window size
- Context is limited to immediate previous words
- May miss important relationships between distant words
- Performance degrades for longer sequences
```

**Final Answer:** The main limitation is that it cannot capture long-range dependencies beyond the fixed window size, limiting its ability to understand context that spans more than 2 words.

**c) Suggest how RNN-based models overcome this limitation (4 marks)**

**Step-by-Step Solution:**

**Step 1: Explain RNN's hidden state mechanism**
```
RNNs maintain a hidden state that carries information from all previous time steps.
```

**Step 2: Describe the information flow**
```
- Hidden state h_t = f(h_{t-1}, x_t) where f is a function
- Information from the entire sequence history is compressed into h_t
- This allows the model to access information from arbitrarily far back
```

**Step 3: Compare with finite memory**
```
- Finite memory: Only looks at fixed window
- RNN: Can theoretically access entire sequence history through hidden state
- RNN: Maintains context across arbitrary distances
```

**Final Answer:** RNNs overcome this limitation by maintaining a hidden state that carries information from all previous time steps, allowing them to capture long-range dependencies across the entire sequence history, not just a fixed window.

### 4. GAN Training Analysis (15 marks)

**Problem Statement:**
You are training a GAN with the following observations:
- Epoch 1-10: Discriminator accuracy = 95%, Generator loss = 5.2
- Epoch 11-20: Discriminator accuracy = 52%, Generator loss = 0.8
- Epoch 21-30: Discriminator accuracy = 50%, Generator loss = 0.7

**a) Analyze what's happening in each phase of training (6 marks)**

**Step-by-Step Solution:**

**Phase 1 (Epochs 1-10):**
```
- Discriminator accuracy: 95% (very high)
- Generator loss: 5.2 (very high)
- Analysis: Discriminator is dominating, easily distinguishing real from fake
- Generator is struggling to produce convincing samples
- This is normal in early training when generator is still learning
```

**Phase 2 (Epochs 11-20):**
```
- Discriminator accuracy: 52% (near optimal)
- Generator loss: 0.8 (significantly reduced)
- Analysis: Balanced competition between generator and discriminator
- Generator has improved significantly and is now fooling discriminator
- This represents the ideal training phase with healthy competition
```

**Phase 3 (Epochs 21-30):**
```
- Discriminator accuracy: 50% (optimal but concerning)
- Generator loss: 0.7 (minimal improvement)
- Analysis: Near-perfect equilibrium, but generator loss plateaued
- Risk of mode collapse or training stagnation
- Generator may be producing limited variety of samples
```

**Final Answer:**
- **Phase 1:** Discriminator dominates, generator learning basics
- **Phase 2:** Balanced competition, optimal training progress
- **Phase 3:** Near equilibrium but potential stagnation/mode collapse

**b) Identify which phase shows the best training progress and why (4 marks)**

**Step-by-Step Solution:**

**Step 1: Evaluate each phase**
```
Phase 1: Poor - discriminator too strong, generator struggling
Phase 2: Good - balanced competition, significant generator improvement
Phase 3: Concerning - plateaued performance, potential issues
```

**Step 2: Analyze the metrics**
```
- Generator loss dropped from 5.2 to 0.8 (84% reduction) in Phase 2
- Discriminator accuracy moved from 95% to 52% (optimal range)
- Phase 3 shows minimal improvement (0.8 to 0.7)
```

**Step 3: Consider training dynamics**
```
- Phase 2 shows healthy adversarial training
- Both networks are learning and adapting
- Generator is successfully challenging discriminator
```

**Final Answer:** **Phase 2 (Epochs 11-20)** shows the best training progress because it demonstrates balanced competition between generator and discriminator, with the generator making significant improvements (84% loss reduction) while the discriminator accuracy reaches the optimal 50% range, indicating healthy adversarial training.

**c) Suggest two techniques to improve training stability (5 marks)**

**Step-by-Step Solution:**

**Technique 1: Learning Rate Scheduling**
```
- Reduce learning rates for both networks
- Use different learning rates for generator and discriminator
- Implement learning rate decay over time
- Helps prevent oscillations and improves convergence
```

**Technique 2: Gradient Penalty (WGAN-GP)**
```
- Add gradient penalty term to discriminator loss
- Prevents discriminator from becoming too strong
- Enforces 1-Lipschitz constraint
- Improves training stability and prevents mode collapse
```

**Alternative Techniques:**
```
- Spectral normalization in discriminator
- Feature matching in generator loss
- Unrolled GANs for better gradient flow
- Progressive growing of networks
```

**Final Answer:**
1. **Learning Rate Scheduling:** Implement adaptive learning rates with decay for both networks to prevent oscillations and improve convergence stability.

2. **Gradient Penalty (WGAN-GP):** Add gradient penalty to discriminator loss to enforce 1-Lipschitz constraint, preventing discriminator from becoming too strong and improving overall training stability.

### 5. VAE Implementation Analysis (15 marks)

**Problem Statement:**
A VAE is trained on MNIST with the following specifications:
- Encoder: 784 → 512 → 256 → 20 (mean and log_var outputs)
- Latent dimension: 10
- Decoder: 10 → 256 → 512 → 784
- β = 1.0 (standard VAE)

**a) Calculate the KL divergence for latent code z = [0.5, -0.2] with μ = [0.3, -0.1] and σ = [0.8, 0.6] (6 marks)**

**Step-by-Step Solution:**

**Step 1: KL divergence formula for VAE**
```
KL(q(z|x) || p(z)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
```

**Step 2: Calculate for first dimension**
```
μ₁ = 0.3, σ₁ = 0.8, z₁ = 0.5
σ₁² = 0.8² = 0.64
log(σ₁²) = log(0.64) = -0.4463

KL₁ = -0.5 × (1 + log(0.64) - 0.3² - 0.64)
KL₁ = -0.5 × (1 - 0.4463 - 0.09 - 0.64)
KL₁ = -0.5 × (-0.1763)
KL₁ = 0.0882
```

**Step 3: Calculate for second dimension**
```
μ₂ = -0.1, σ₂ = 0.6, z₂ = -0.2
σ₂² = 0.6² = 0.36
log(σ₂²) = log(0.36) = -1.0217

KL₂ = -0.5 × (1 + log(0.36) - (-0.1)² - 0.36)
KL₂ = -0.5 × (1 - 1.0217 - 0.01 - 0.36)
KL₂ = -0.5 × (-0.3917)
KL₂ = 0.1959
```

**Step 4: Total KL divergence**
```
Total KL = KL₁ + KL₂ = 0.0882 + 0.1959 = 0.2841
```

**Final Answer: KL divergence = 0.2841**

**b) If generated images are blurry, what could be the problem and how would you fix it? (5 marks)**

**Step-by-Step Solution:**

**Step 1: Identify potential causes of blurry images**
```
1. KL divergence term too strong (β too high)
2. Latent dimension too small
3. Decoder architecture insufficient
4. Training data preprocessing issues
5. Loss function weighting problems
```

**Step 2: Analyze the specific case**
```
- β = 1.0 (standard VAE)
- Latent dimension = 10 (reasonable for MNIST)
- Architecture seems adequate
```

**Step 3: Most likely causes and solutions**
```
Problem 1: KL divergence dominating reconstruction loss
Solution: Reduce β to 0.1-0.5 to prioritize reconstruction quality

Problem 2: Decoder capacity insufficient
Solution: Increase decoder layers or neurons, add skip connections

Problem 3: Training instability
Solution: Use better optimizer (Adam), learning rate scheduling
```

**Final Answer:**
**Most likely problem:** KL divergence term is too strong, causing the model to prioritize latent space regularization over reconstruction quality.

**Solutions:**
1. **Reduce β parameter** to 0.1-0.5 to decrease KL divergence weight
2. **Enhance decoder architecture** with more layers or skip connections
3. **Use better optimization** with Adam optimizer and learning rate scheduling

**c) How would you modify the loss function to improve disentanglement? (4 marks)**

**Step-by-Step Solution:**

**Step 1: Understand disentanglement goal**
```
Disentanglement: Each latent dimension should control one interpretable factor
Current VAE: Latent dimensions may be correlated
```

**Step 2: β-VAE approach**
```
Modify loss: L = Reconstruction_loss + β × KL_divergence
Increase β > 1 (e.g., β = 4-10) to encourage more independent latent factors
```

**Step 3: β-TC-VAE approach**
```
Add total correlation penalty:
L = Reconstruction_loss + β × KL_divergence + γ × Total_Correlation
Total_Correlation = KL(q(z) || ∏ᵢ q(zᵢ))
```

**Step 4: Factor-VAE approach**
```
Add factor-wise KL divergence:
L = Reconstruction_loss + β × KL_divergence + γ × Factor_KL
Factor_KL penalizes correlation between latent dimensions
```

**Final Answer:**
**Modify loss function to improve disentanglement:**

1. **β-VAE:** Increase β to 4-10: `L = Reconstruction_loss + β × KL_divergence`
2. **β-TC-VAE:** Add total correlation penalty: `L = Reconstruction_loss + β × KL_divergence + γ × Total_Correlation`
3. **Factor-VAE:** Add factor-wise regularization to encourage independent latent factors

### 6. Convolutional Architecture Design (15 marks)

**Problem Statement:**
Design a CNN for generating 32×32 RGB images using the following constraints:
- Start with 100-dimensional noise vector
- Use transpose convolutions for upsampling
- Include batch normalization
- Final output should be in range [-1, 1]

**a) Design the complete architecture with layer specifications (10 marks)**

**Step-by-Step Solution:**

**Step 1: Determine upsampling strategy**
```
Target: 32×32×3
Start: 100-dimensional vector
Need to upsample: 100 → 32×32×3 = 3,072
```

**Step 2: Design progressive upsampling**
```
100 → 4×4×256 → 8×8×128 → 16×16×64 → 32×32×3
```

**Step 3: Detailed architecture design**
```
Layer 1: Fully Connected
- Input: 100
- Output: 4×4×256 = 4,096
- Activation: ReLU
- Batch Normalization: Yes

Layer 2: Reshape
- Input: 4,096
- Output: 4×4×256

Layer 3: ConvTranspose2d
- Input: 4×4×256
- Output: 8×8×128
- Kernel: 4×4, Stride: 2, Padding: 1
- Activation: ReLU
- Batch Normalization: Yes

Layer 4: ConvTranspose2d
- Input: 8×8×128
- Output: 16×16×64
- Kernel: 4×4, Stride: 2, Padding: 1
- Activation: ReLU
- Batch Normalization: Yes

Layer 5: ConvTranspose2d
- Input: 16×16×64
- Output: 32×32×3
- Kernel: 4×4, Stride: 2, Padding: 1
- Activation: Tanh (for [-1,1] range)
- Batch Normalization: No (final layer)
```

**Final Answer:**
```
1. FC: 100 → 4,096 (ReLU + BatchNorm)
2. Reshape: 4,096 → 4×4×256
3. ConvT: 4×4×256 → 8×8×128 (4×4 kernel, stride=2, ReLU + BatchNorm)
4. ConvT: 8×8×128 → 16×16×64 (4×4 kernel, stride=2, ReLU + BatchNorm)
5. ConvT: 16×16×64 → 32×32×3 (4×4 kernel, stride=2, Tanh)
```

**b) Calculate the number of parameters in each layer (5 marks)**

**Step-by-Step Solution:**

**Step 1: FC Layer parameters**
```
Weights: 100 × 4,096 = 409,600
Bias: 4,096
Total: 409,600 + 4,096 = 413,696
```

**Step 2: ConvT Layer 1 parameters**
```
Weights: 4×4×256×128 = 524,288
Bias: 128
Total: 524,288 + 128 = 524,416
```

**Step 3: ConvT Layer 2 parameters**
```
Weights: 4×4×128×64 = 131,072
Bias: 64
Total: 131,072 + 64 = 131,136
```

**Step 4: ConvT Layer 3 parameters**
```
Weights: 4×4×64×3 = 3,072
Bias: 3
Total: 3,072 + 3 = 3,075
```

**Step 5: Total parameters**
```
Total = 413,696 + 524,416 + 131,136 + 3,075 = 1,072,323
```

**Final Answer:**
- FC Layer: 413,696 parameters
- ConvT Layer 1: 524,416 parameters
- ConvT Layer 2: 131,136 parameters
- ConvT Layer 3: 3,075 parameters
- **Total: 1,072,323 parameters**

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
