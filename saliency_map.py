with tf.GradientTape() as tape:
    tape.watch(input_img)
    result = test_model(input_img)
    max_score = result[0,max_idx[0]]
grads = tape.gradient(max_score, input_img)


plot_maps(normalize_image(grads[0]), normalize_image(input_img[0]))
