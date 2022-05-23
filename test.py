test_model = tf.keras.applications.resnet50.ResNet50()
#test_model.summary()
get_image()
img_path = "image.jpg"
input_img = input_img(img_path)
input_img = tf.keras.applications.densenet.preprocess_input(input_img)
plt.imshow(normalize_image(input_img[0]), cmap = "ocean")
