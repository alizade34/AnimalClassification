
# Image Classification Project

This project demonstrates how to build and train Convolutional Neural Networks (CNN) and ResNet50V2 for image classification using TensorFlow and Keras.

## Requirements

- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- scikit-learn

You can install these packages using the following command:

```sh
pip install -r requirements.txt
```

## Project Structure

The main steps in the project are:

1. **Creating a Sampled Dataset**
2. **Data Augmentation and Preprocessing**
3. **Building and Training Models**
4. **Evaluating Models**
5. **Plotting Results**

## Dataset

- The dataset should be structured in the following format:
  ```
  ├── class1
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── class2
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...
  ```

- Use the `create_sampled_dataset` function to create a smaller subset of the dataset if needed.

## Usage

### Step 1: Creating a Sampled Dataset

Uncomment the following lines to create a sampled dataset:

```python
src_dir = 'raw-img'
dest_dir = 'raw-img-sampled'
create_sampled_dataset(src_dir, dest_dir, num_samples=1000)
```

### Step 2: Data Augmentation and Preprocessing

Define the path to your dataset and set parameters for the image data generator:

```python
data_path = 'path_to_your_dataset'
batch_size = 32
img_size = 224
```

### Step 3: Building and Training Models

#### Custom CNN

Build and train a custom CNN model:

```python
model = build_custom_cnn(img_size, num_classes)
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint_cnn, earlystopping, learning_rate_reduction]
)
```

#### ResNet50V2

Build and train a ResNet50V2 model:

```python
model = build_resnet50v2(img_size, num_classes)
history_resnet = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint_resnet50v2, earlystopping, learning_rate_reduction]
)
```

### Step 4: Evaluating Models

Evaluate the model and generate a classification report and confusion matrix:

```python
validation_generator.reset()
preds = model.predict(validation_generator)
pred_classes = np.argmax(preds, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

report = classification_report(true_classes, pred_classes, target_names=class_labels)
print(report)

conf_matrix = confusion_matrix(true_classes, pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.show()
```

### Step 5: Plotting Results

Plot the training and validation loss and accuracy:

```python
plot_history(history)  # For custom CNN
plot_history_resnet(history_resnet)  # For ResNet50V2
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
