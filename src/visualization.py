import matplotlib.pyplot as plt

def show_image_grid(images, rows, cols, height, width, title, subtitles):
	plt.close()
	fig, ax_arr = plt.subplots(rows, cols)
	fig.set_size_inches(width, height)
	plt.suptitle(title, fontsize=22)
	plt.tight_layout(rect=(0, 0, 1, 0.85))
	i = 0
	for x in range(rows):
		for y in range(cols):
			ax_arr[x*cols + y].imshow(images[i, ...])
			ax_arr[x*cols + y].set_title(subtitles[i], fontsize=16)
			i += 1
	plt.show()
	plt.close()