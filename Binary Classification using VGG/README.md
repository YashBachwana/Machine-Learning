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
