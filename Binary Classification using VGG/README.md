# Apple vs Banana Classification

This project aims to compare the performance of various models on a binary classification task to distinguish between images of apples and bananas. The models being evaluated include different configurations of the VGG architecture, as well as transfer learning approaches.

## Model Architectures

We evaluated the following models:

1. **VGG (1 block)**
2. **VGG (3 blocks)**
3. **VGG (3 blocks) with Data Augmentation**
4. **Transfer Learning using VGG16/VGG19 (Tuning All Layers)**
5. **Transfer Learning using VGG16/VGG19 (Tuning MLP Layers Only)**

### Model Implementations

```python
def create_model_vgg1():
    model = Sequential([
        Input(shape=(224, 224, 3)),  # Specify input shape here
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    opt = SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
```
    
```python
def create_model_vgg3():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    opt = SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

### Data Augmentation
We utilized the following data augmentation techniques to enhance the training dataset:
    
```python
data_gen = ImageDataGenerator(
    rotation_range=40,       
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    fill_mode='nearest'      
)
```

### Transfer Learning with VGG16 
1. All Layers Tuned
```python
model_transfer_all = Sequential([
    base_model,
    Flatten(),  # Use global average pooling to reduce the spatial dimensions
    Dense(512, activation='relu'),  # A dense layer as a hidden layer
    Dense(256, activation='relu'),  # A dense layer as a hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])
```
2. Only MLP Layers Tuned
```python
model_transfer_mlp = Sequential([
    base_model_mlp,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
### Comparative Analysis

| Model                                    | Train Time (mins) | Train Loss | Train Acc (%) | Test Acc (%) | Params (M)   |
|------------------------------------------|--------------------|------------|----------------|---------------|---------------|
| VGG (1 block)                           | 0.20               | 0.2174     | 92.66          | 88.89         | 50,467,969    |
| VGG (3 blocks)                          | 1.07               | 0.1627     | 95.48          | 91.11         | 11,169,089    |
| VGG (3 blocks) with augmentation        | 1.00               | 0.0405     | 98.68          | 97.78         | 11,169,089    |
| Transfer Learning (VGG16, all layers)  | 19.69              | 0.0190     | 100.00         | 100.00        | 27,691,841    |
| Transfer Learning (VGG16, MLP only)    | 6.96               | 0.0012     | 100.00         | 100.00        | 21,137,729    |
| MLP Model                                | 0.05               | 0.4922     | 75.71          | 73.33         | 0.01          |


